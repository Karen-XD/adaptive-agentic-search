# third_party

## Search-R1

- Upstream: https://github.com/PeterGriffinJin/Search-R1
- Pinned commit: `598e61bd1d36895726d28a8d06b3a15bed19f5d3` (2025-11-13, "Merge pull request #164")
- Included as a git submodule; the upstream tree is kept unmodified.

Local changes live in `patches/` and are applied on demand:

```bash
git submodule update --init
git -C third_party/Search-R1 apply ../patches/*.patch
```

| Patch | Purpose |
|---|---|
| `0001-merge-scripts-local-jsonl-per-data-source.patch` | `qa_search_{train,test}_merge.py`: load `{raw_dir}/{data_source}_{split}.jsonl` when present, fall back to `RUC-NLPIR/FlashRAG_datasets` |
