import os
import sys
import types
import unittest
from unittest.mock import patch

from generative_agent.ai_analysis import FinalAnalysisConfig, generate_final_analysis


class AiAnalysisTests(unittest.TestCase):
    def test_auto_mode_is_transparent_without_credentials(self):
        with patch.dict(os.environ, {}, clear=True):
            result = generate_final_analysis({"selected": "baseline"})
        self.assertEqual(result["status"], "fallback")
        self.assertIn("configured AI provider", result["reason"])

    def test_disabled_mode_does_not_call_provider(self):
        result = generate_final_analysis({}, FinalAnalysisConfig(mode="disabled"))
        self.assertEqual(result["status"], "disabled")

    def test_openai_result_is_recorded_without_changing_evidence(self):
        class Response:
            output_text = "The baseline ranked highest on validation; use only for internal testing."

        class Client:
            class responses:
                @staticmethod
                def create(**kwargs):
                    return Response()

        fake_openai = types.SimpleNamespace(OpenAI=lambda: Client())
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}), patch.dict(
            sys.modules, {"openai": fake_openai}
        ):
            result = generate_final_analysis(
                {"selected": "baseline"}, FinalAnalysisConfig(mode="openai", model="test-model")
            )
        self.assertEqual(result["status"], "generated")
        self.assertEqual(result["model"], "test-model")
        self.assertIn("baseline", result["text"])

    def test_anthropic_result_is_recorded_without_exposing_credentials(self):
        class Block:
            type = "text"
            text = "The selected result is supported for internal testing only."

        class Response:
            content = [Block()]

        class Client:
            class messages:
                @staticmethod
                def create(**kwargs):
                    return Response()

        fake_anthropic = types.SimpleNamespace(Anthropic=lambda: Client())
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}), patch.dict(
            sys.modules, {"anthropic": fake_anthropic}
        ):
            result = generate_final_analysis(
                {"selected": "baseline"},
                FinalAnalysisConfig(mode="anthropic", model="claude-haiku-5-5"),
            )
        self.assertEqual(result["status"], "generated")
        self.assertEqual(result["provider"], "anthropic")
        self.assertNotIn("test-key", str(result))


if __name__ == "__main__":
    unittest.main()
