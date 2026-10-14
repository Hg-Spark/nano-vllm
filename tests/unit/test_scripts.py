"""CLI behavior checks that run without importing the GPU runtime."""

import contextlib
import io
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from scripts import chat
from scripts._common import compare_tokens, run_nano_greedy


class TokenComparisonTest(unittest.TestCase):
    def test_nano_greedy_configuration_ignores_eos(self):
        engine = SimpleNamespace(
            generate=Mock(return_value=[{"token_ids": [99, 3, 4]}]),
        )

        class Sampling:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        with patch.dict(
            "sys.modules",
            {"nanovllm": SimpleNamespace(SamplingParams=Sampling)},
        ):
            self.assertEqual(run_nano_greedy(engine, "prompt", 3), [99, 3, 4])
        engine.generate.assert_called_once()
        args, kwargs = engine.generate.call_args
        self.assertEqual(args[0], ["prompt"])
        self.assertEqual(args[1].temperature, 0.0)
        self.assertEqual(args[1].max_tokens, 3)
        self.assertTrue(args[1].ignore_eos)
        self.assertIs(kwargs["use_tqdm"], False)

    def test_token_comparison_reports_token_and_length_mismatches(self):
        for left, right, expected in (
            ([], [], None),
            ([1, 2], [1, 2], None),
            ([1, 2], [1, 3], 1),
            ([1], [1, 2], 1),
            ([1, 2], [1], 1),
            ([], [1], 0),
            ([1], [], 0),
        ):
            with self.subTest(left=left, right=right):
                self.assertEqual(
                    compare_tokens(left, right),
                    {
                        "matched": expected is None,
                        "reference_length": len(left),
                        "actual_length": len(right),
                        "first_mismatch": expected,
                        "reference_token": (
                            left[expected]
                            if expected is not None and expected < len(left)
                            else None
                        ),
                        "actual_token": (
                            right[expected]
                            if expected is not None and expected < len(right)
                            else None
                        ),
                    },
                )


def make_chat_engine():
    # Transformers 5.x defaults to BatchEncoding, not a plain token list.
    def apply_chat_template(messages, *, return_dict=True, **kwargs):
        return {"input_ids": [10, 20]} if return_dict else [10, 20]

    return SimpleNamespace(
        tokenizer=SimpleNamespace(
            apply_chat_template=Mock(side_effect=apply_chat_template),
            decode=Mock(return_value="clean reply"),
        ),
        generate=Mock(return_value=[{"token_ids": [30, 99]}]),
        is_finished=Mock(return_value=True),
        exit=Mock(),
    )


class ChatTest(unittest.TestCase):
    def run_dialogue(self, engine, inputs, system=""):
        output = io.StringIO()
        with patch("builtins.input", side_effect=inputs), contextlib.redirect_stdout(output):
            chat.run_chat(engine, object(), system_prompt=system)
        return output.getvalue()

    def test_multiple_turns_reuse_clean_assistant_history_and_token_ids(self):
        engine = make_chat_engine()
        self.run_dialogue(engine, ["hello", "continue", "/exit"], "system")
        second = engine.tokenizer.apply_chat_template.call_args_list[1]
        self.assertEqual(
            second.args[0],
            [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "clean reply"},
                {"role": "user", "content": "continue"},
            ],
        )
        self.assertEqual(
            second.kwargs,
            {
                "tokenize": True,
                "add_generation_prompt": True,
                "return_dict": False,
            },
        )
        self.assertEqual(engine.generate.call_args.args[0], [[10, 20]])
        engine.tokenizer.decode.assert_called_with([30, 99], skip_special_tokens=True)

    def test_clear_preserves_system_and_help_does_not_generate(self):
        engine = make_chat_engine()
        self.run_dialogue(engine, ["hello", "/clear", " ", "/help", "new", "/quit"], "system")
        self.assertEqual(engine.generate.call_count, 2)
        self.assertEqual(
            engine.tokenizer.apply_chat_template.call_args.args[0],
            [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "new"},
            ],
        )

    def test_rejected_request_does_not_pollute_history(self):
        engine = make_chat_engine()
        engine.generate.side_effect = [
            [{"token_ids": [30, 99]}],
            ValueError("requested prompt + completion exceeds max_model_len"),
            [{"token_ids": [30, 99]}],
        ]
        output = self.run_dialogue(engine, ["hello", "too long", "short", "/exit"])
        self.assertIn("Request rejected:", output)
        self.assertEqual(
            engine.tokenizer.apply_chat_template.call_args.args[0],
            [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "clean reply"},
                {"role": "user", "content": "short"},
            ],
        )

    def test_active_engine_failure_is_not_treated_as_input_rejection(self):
        engine = make_chat_engine()
        engine.generate.side_effect = ValueError("execution failed")
        engine.is_finished.return_value = False
        with self.assertRaisesRegex(ValueError, "execution failed"):
            self.run_dialogue(engine, ["hello"])

    def test_eof_and_ctrl_c_exit_input_loop(self):
        for interrupt in (EOFError, KeyboardInterrupt):
            with self.subTest(interrupt=interrupt):
                engine = make_chat_engine()
                self.run_dialogue(engine, [interrupt])
                engine.generate.assert_not_called()

    def test_main_closes_engine_on_failed_or_interrupted_generation(self):
        for failure in (RuntimeError("generation failed"), KeyboardInterrupt()):
            with self.subTest(failure=type(failure).__name__):
                engine = make_chat_engine()
                engine.generate.side_effect = failure
                llm_factory = Mock(return_value=engine)
                sampling_factory = Mock(return_value=object())
                runtime = SimpleNamespace(LLM=llm_factory, SamplingParams=sampling_factory)
                with (
                    patch.dict("sys.modules", {"nanovllm": runtime}),
                    patch("builtins.input", return_value="hello"),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    if isinstance(failure, KeyboardInterrupt):
                        chat.main(
                            [
                                "model",
                                "--dtype",
                                "float16",
                                "--decode-graph",
                            ]
                        )
                    else:
                        with self.assertRaisesRegex(RuntimeError, "generation failed"):
                            chat.main(
                                [
                                    "model",
                                    "--dtype",
                                    "float16",
                                    "--decode-graph",
                                ]
                            )
                engine.exit.assert_called_once()
                self.assertEqual(llm_factory.call_args.kwargs["decode_graph_batch_sizes"], (1,))
                self.assertEqual(llm_factory.call_args.kwargs["dtype"], "float16")
                self.assertFalse(sampling_factory.call_args.kwargs["ignore_eos"])


if __name__ == "__main__":
    unittest.main()
