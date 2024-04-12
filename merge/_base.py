from peft import PeftModel


class Merger:
    def __init__(
        self,
        baseline_model,
        finetuned_model_pth,
        tokenizer,
        name="merged_finetuned_mistral",
    ) -> None:
        self._baseline_model = baseline_model
        self._finetuned_model_pth = finetuned_model_pth
        self._tokenizer = tokenizer
        self._merged_model_name = name

    def merge(self) -> None:
        self._merged_model = PeftModel.from_pretrained(
            self._baseline_model, self._finetuned_model_pth
        )
        self._merged_model.merge_and_unload()
        self._merged_model.save_pretrained(
            self._merged_model_name, safe_serialization=True, max_shard_size="2GB"
        )
        self._tokenizer.save_pretrained(self._merged_model_name)
