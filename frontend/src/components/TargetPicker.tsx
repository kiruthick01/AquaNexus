/**
 * Choosing which model answers.
 *
 * The synthetic model is selectable - scenario work needs a bounded habitat
 * score - but picking it changes the surrounding copy, not just a query
 * parameter. If the two models looked interchangeable here, nothing downstream
 * could undo that impression.
 */

import type { ModelInfo, Target } from "../types";

const LABELS: Record<Target, string> = {
  dissolved_oxygen: "Dissolved oxygen",
  hsi: "Habitat index",
};

export default function TargetPicker({
  value,
  onChange,
  models,
  disabled,
}: {
  value: Target;
  onChange: (target: Target) => void;
  models: ModelInfo[];
  disabled?: boolean;
}) {
  const targets: Target[] = ["dissolved_oxygen", "hsi"];
  const selected = models.find((model) => model.target === value);

  return (
    <div>
      <div className="segmented" role="group" aria-label="Model">
        {targets.map((target) => (
          <button
            key={target}
            type="button"
            aria-pressed={value === target}
            disabled={disabled}
            onClick={() => onChange(target)}
          >
            {LABELS[target]}
          </button>
        ))}
      </div>
      {selected && (
        <p className="small muted" style={{ marginTop: "0.5rem" }}>
          {selected.model_type} · trained on {selected.n_train.toLocaleString()}{" "}
          {selected.labels === "SYNTHETIC" ? "generated" : "measured"} rows
          {typeof selected.metrics.r2 === "number" &&
            ` · R² ${(selected.metrics.r2 as number).toFixed(2)}`}
          {selected.labels === "SYNTHETIC" &&
            " — a score that measures recovery of a function this project wrote"}
        </p>
      )}
    </div>
  );
}
