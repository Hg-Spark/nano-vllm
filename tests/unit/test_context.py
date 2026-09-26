import unittest

from nanovllm.utils.context import Context, get_context, use_context


class ContextScopeTest(unittest.TestCase):

    def test_nested_scope_restores_previous_context_after_error(self):
        original = get_context()
        outer = Context(is_prefill=True)
        inner = Context(is_prefill=False)

        with use_context(outer):
            self.assertIs(get_context(), outer)
            with self.assertRaisesRegex(RuntimeError, "injected"):
                with use_context(inner):
                    self.assertIs(get_context(), inner)
                    raise RuntimeError("injected")
            self.assertIs(get_context(), outer)

        self.assertIs(get_context(), original)


if __name__ == "__main__":
    unittest.main()
