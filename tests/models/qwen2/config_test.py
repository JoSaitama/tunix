"""DeepSeek architecture compatibility without allocating model weights."""

from absl.testing import absltest
from tunix.models import automodel


class DeepSeekConfigTest(absltest.TestCase):

  def test_7b_hf_id_routes_to_deepseek_architecture(self):
    cfg = automodel.call_model_config("deepseek-r1-distill-qwen-7b")
    self.assertEqual(
        (cfg.num_layers, cfg.vocab_size, cfg.embed_dim, cfg.hidden_dim),
        (28, 152064, 3584, 18944),
    )
    self.assertEqual((cfg.num_heads, cfg.num_kv_heads, cfg.head_dim), (28, 4, 128))
    self.assertEqual(cfg.rope_theta, 10000)
    self.assertFalse(cfg.use_tied_embedding)
    # Accidentally selecting the generic Qwen config silently changes RoPE.
    qwen_cfg = automodel.call_model_config("qwen2.5-7b")
    self.assertEqual(qwen_cfg.rope_theta, 1_000_000)

  def test_1p5b_architecture_still_matches_existing_checkpoints(self):
    cfg = automodel.call_model_config("deepseek-r1-distill-qwen-1.5b")
    self.assertEqual(
        (cfg.num_layers, cfg.vocab_size, cfg.embed_dim, cfg.hidden_dim),
        (28, 151936, 1536, 8960),
    )
    self.assertEqual((cfg.num_heads, cfg.num_kv_heads, cfg.head_dim), (12, 2, 128))
    self.assertEqual(cfg.rope_theta, 10000)
    self.assertFalse(cfg.use_tied_embedding)


if __name__ == "__main__":
  absltest.main()
