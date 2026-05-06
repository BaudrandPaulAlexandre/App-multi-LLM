"""
test_prompting.py
-----------------
Tests des stratégies de prompting (Lot A baseline + Lot C variantes).

Couvre :
  - VanillaStrategy        : fidélité au texte brut
  - SystemPromptStrategy   : preset, custom, validation
  - PrefixSuffixStrategy   : par langue, fallback, override
  - RewriteStrategy        : succès, fallback en cas d'erreur du rewriter
  - build_strategy()       : factory + erreurs
"""

from unittest.mock import MagicMock

import pytest

from eloquent.prompting import (
    PrefixSuffixStrategy,
    PromptBuildResult,
    RewriteStrategy,
    SystemPromptStrategy,
    VanillaStrategy,
    build_strategy,
)
from eloquent.providers import LLMResponse


# ---------------------------------------------------------------------------
# Vanilla
# ---------------------------------------------------------------------------

class TestVanilla:

    def test_messages_contain_only_user_role(self):
        result = VanillaStrategy().build("Quelle est la capitale ?", "fr")
        assert len(result.messages) == 1
        assert result.messages[0]["role"] == "user"
        assert result.messages[0]["content"] == "Quelle est la capitale ?"

    def test_trace_records_strategy(self):
        result = VanillaStrategy().build("test", "en")
        assert result.trace == {"strategy": "vanilla"}


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

class TestSystemPrompt:

    def test_preset_concise(self):
        result = SystemPromptStrategy(preset="concise").build("Q ?", "fr")
        assert result.messages[0]["role"] == "system"
        assert "concise" in result.messages[0]["content"].lower()
        assert result.messages[1] == {"role": "user", "content": "Q ?"}
        assert result.trace["preset"] == "concise"

    def test_custom_system_prompt_overrides_preset(self):
        result = SystemPromptStrategy(
            system_prompt="Be funny.", preset="concise",
        ).build("Q ?", "fr")
        assert result.messages[0]["content"] == "Be funny."
        assert result.trace["preset"] == "custom"

    def test_raises_when_neither_preset_nor_prompt_given(self):
        with pytest.raises(ValueError, match="requiert"):
            SystemPromptStrategy()

    def test_unknown_preset_falls_through_to_error(self):
        with pytest.raises(ValueError):
            SystemPromptStrategy(preset="does_not_exist")


# ---------------------------------------------------------------------------
# Prefix / suffix
# ---------------------------------------------------------------------------

class TestPrefixSuffix:

    def test_default_french_prefix_and_suffix(self):
        result = PrefixSuffixStrategy().build("Quelle heure ?", "fr")
        content = result.messages[0]["content"]
        assert content.startswith("Réponds en une seule phrase courte, en français : ")
        assert "Quelle heure ?" in content
        assert content.endswith("Réponse (une phrase) :")

    def test_unknown_language_yields_no_prefix_or_suffix(self):
        # Le polonais n'est pas dans la table par défaut → texte intact
        result = PrefixSuffixStrategy().build("Pytanie ?", "pl")
        assert result.messages[0]["content"] == "Pytanie ?"
        assert result.trace["prefix"] == ""
        assert result.trace["suffix"] == ""

    def test_user_override_merges_with_defaults(self):
        # On override seulement le préfixe FR : EN garde les défauts
        strat = PrefixSuffixStrategy(
            prefixes={"fr": "PFX_FR "},
        )
        fr = strat.build("Q", "fr").messages[0]["content"]
        en = strat.build("Q", "en").messages[0]["content"]
        assert fr.startswith("PFX_FR ")
        assert en.startswith("Answer in one short sentence")  # défaut conservé

    def test_trace_contains_lang_prefix_suffix(self):
        result = PrefixSuffixStrategy().build("Q", "es")
        assert result.trace["lang"] == "es"
        assert "español" in result.trace["prefix"]


# ---------------------------------------------------------------------------
# Rewrite
# ---------------------------------------------------------------------------

class TestRewrite:

    def _make_rewriter(self, content: str = "", error: str | None = None):
        rewriter = MagicMock()
        rewriter.provider_name = "mock_rewriter"
        rewriter.generate_safe.return_value = LLMResponse(
            content=content,
            model="mock",
            provider_name="mock_rewriter",
            latency_ms=42.0,
            error=error,
        )
        return rewriter

    def test_uses_rewritten_text_when_rewriter_succeeds(self):
        rewriter = self._make_rewriter(content="Quelle est la capitale de la France ?")
        result = RewriteStrategy(rewriter=rewriter).build(
            "C'est quoi la cap?", "fr",
        )
        # Le LLM cible reçoit la version réécrite, pas l'originale
        assert result.messages[0]["content"] == "Quelle est la capitale de la France ?"
        assert result.trace["rewriter_status"] == "ok"
        assert result.trace["original_text"] == "C'est quoi la cap?"
        assert result.trace["rewritten_text"] == "Quelle est la capitale de la France ?"

    def test_falls_back_to_original_when_rewriter_errors(self):
        rewriter = self._make_rewriter(content="", error="rate limit exceeded")
        result = RewriteStrategy(rewriter=rewriter).build("Question ?", "fr")
        assert result.messages[0]["content"] == "Question ?"
        assert result.trace["rewriter_status"] == "fallback_original"

    def test_falls_back_when_rewriter_returns_blank(self):
        rewriter = self._make_rewriter(content="   ")
        result = RewriteStrategy(rewriter=rewriter).build("Question ?", "fr")
        assert result.messages[0]["content"] == "Question ?"
        assert result.trace["rewriter_status"] == "fallback_original"


# ---------------------------------------------------------------------------
# Factory build_strategy()
# ---------------------------------------------------------------------------

class TestBuildStrategy:

    def test_vanilla(self):
        assert build_strategy("vanilla").strategy_name == "vanilla"

    def test_system_prompt_with_preset(self):
        s = build_strategy("system_prompt", preset="neutral")
        assert s.strategy_name == "system_prompt"

    def test_prefix_suffix(self):
        assert build_strategy("prefix_suffix").strategy_name == "prefix_suffix"

    def test_rewrite_requires_rewriter(self):
        with pytest.raises(ValueError, match="rewriter"):
            build_strategy("rewrite")

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="Stratégie inconnue"):
            build_strategy("not_a_strategy")
