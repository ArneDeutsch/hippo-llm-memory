# Local model fixtures

The previous `models/tiny-gpt2` checkpoint has been replaced by the
synthetic `hippo/fake-tiny-gpt2` backend provided by
`hippo_mem.testing.fake_hf`. The directory is kept so tooling that
expects a `models/` folder can continue to symlink against the repository
root without downloading checkpoints.
