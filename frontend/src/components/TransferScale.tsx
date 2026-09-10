/**
 * Where the transfer lands, drawn against the datums that give it meaning.
 *
 * An R² on its own says nothing. Two references do: **zero**, which is what
 * predicting this river's own mean scores and needs no model at all, and the
 * score the same model gets on the river it was trained on. Drawing all three
 * on one staff is the whole argument of the held-out test — inside its evidence
 * the model sits beside its home score, outside it falls off the left of the
 * scale, below the line where a model stops being worth having.
 *
 * A table of the same four numbers would let a reader take the average. The
 * staff makes the distance between the two regimes the thing you see first.
 */

import type { TransferScore } from "../types";

const WIDTH = 560;
const HEIGHT = 146;
const PAD = 16;
const AXIS_Y = 112;
const CEILING = 0.6;
/** Datum labels sit in rows across the top, clear of the needles below. */
const DATUM_ROWS = [26, 46];

/** Where a score sits on the staff. */
function scale(value: number, floor: number): number {
  const span = CEILING - floor;
  return PAD + ((value - floor) / span) * (WIDTH - 2 * PAD);
}

export interface Mark {
  label: string;
  detail?: string;
  r2: number;
  colour: string;
  /** Dashed marks are references, not results of the transfer. */
  reference?: boolean;
  /** Vertical stacking, so two nearby marks do not overprint. */
  level: number;
}

export default function TransferScale({
  marks,
  floor,
}: {
  marks: Mark[];
  floor: number;
}) {
  const ticks = tickValues(floor);
  const references = marks.filter((mark) => mark.reference);
  const results = marks.filter((mark) => !mark.reference);

  return (
    <figure className="transfer-scale">
      <svg
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        width="100%"
        role="img"
        aria-label={marks.map((mark) => `${mark.label}: R² ${signed(mark.r2)}`).join("; ")}
      >
        {/* The staff. */}
        <line
          x1={PAD}
          y1={AXIS_Y}
          x2={WIDTH - PAD}
          y2={AXIS_Y}
          stroke="var(--rule-strong)"
          strokeWidth={1}
        />

        {ticks.map((value) => (
          <g key={value}>
            <line
              x1={scale(value, floor)}
              y1={AXIS_Y}
              x2={scale(value, floor)}
              y2={AXIS_Y + 5}
              stroke="var(--rule-strong)"
              strokeWidth={1}
            />
            <text
              x={scale(value, floor)}
              y={AXIS_Y + 18}
              textAnchor="middle"
              fontSize={11}
              fill="var(--ink-3)"
              fontFamily="var(--mono)"
            >
              {value === 0 ? "0" : value.toFixed(1)}
            </text>
          </g>
        ))}

        {/* Zero is the datum: a model that scores it has earned nothing. */}
        <line
          x1={scale(0, floor)}
          y1={DATUM_ROWS[0] + 2}
          x2={scale(0, floor)}
          y2={AXIS_Y}
          stroke="var(--ink)"
          strokeWidth={1}
        />
        <DatumLabel
          x={scale(0, floor)}
          y={DATUM_ROWS[0]}
          text="this river's own mean"
        />

        {/*
         * A reference is not a result of the transfer, so it is drawn the way
         * zero is - a rule across the plot with its label at the top - rather
         * than as a needle. Drawn as a needle it landed 0.06 from the in-range
         * mark and its leader line crossed that mark's own label.
         */}
        {references.map((mark, index) => {
          const x = scale(mark.r2, floor);
          const row = DATUM_ROWS[Math.min(index + 1, DATUM_ROWS.length - 1)];
          return (
            <g key={mark.label}>
              <line
                x1={x}
                y1={row + 2}
                x2={x}
                y2={AXIS_Y}
                stroke={mark.colour}
                strokeWidth={1}
                strokeDasharray="3 3"
              />
              <circle cx={x} cy={AXIS_Y} r={3} fill={mark.colour} />
              <DatumLabel x={x} y={row} text={`${mark.label} ${signed(mark.r2)}`} />
            </g>
          );
        })}

        {results.map((mark) => {
          const x = scale(mark.r2, floor);
          const y = AXIS_Y - 18 - mark.level * 34;
          const anchor = x < 90 ? "start" : x > WIDTH - 90 ? "end" : "middle";
          const nudge = anchor === "start" ? -6 : anchor === "end" ? 6 : 0;

          return (
            <g key={mark.label}>
              <line
                x1={x}
                y1={y + 6}
                x2={x}
                y2={AXIS_Y}
                stroke={mark.colour}
                strokeWidth={2}
              />
              <circle cx={x} cy={AXIS_Y} r={4.5} fill={mark.colour} />
              <text
                x={x + nudge}
                y={y}
                textAnchor={anchor}
                fontSize={13}
                fontFamily="var(--mono)"
                fill={mark.colour}
                fontWeight={600}
              >
                {signed(mark.r2)}
              </text>
              <text
                x={x + nudge}
                y={y - 14}
                textAnchor={anchor}
                fontSize={11}
                fill="var(--ink-2)"
              >
                {mark.label}
              </text>
            </g>
          );
        })}
      </svg>
      <figcaption>
        R² on the held-out river. Anything at or below the datum is beaten by
        predicting that river's mean, which needs no model, no geometry and no
        simulation.
      </figcaption>
    </figure>
  );
}

/**
 * A datum's label, set on whichever side of its rule has room.
 *
 * Always setting it to the right ran the home reference off the plot, because
 * that rule stands near the right-hand end of the staff — which is the whole
 * point of where it stands.
 */
function DatumLabel({ x, y, text }: { x: number; y: number; text: string }) {
  const width = text.length * 5.4; // 11px sans, near enough to choose a side
  const flip = x + width > WIDTH - PAD;
  return (
    <text
      x={flip ? x - 6 : x + 6}
      y={y}
      textAnchor={flip ? "end" : "start"}
      fontSize={11}
      fill="var(--ink-2)"
    >
      {text}
    </text>
  );
}

/** R² is a signed quantity; an unsigned 0.336 reads as a magnitude. */
function signed(value: number): string {
  return `${value >= 0 ? "+" : ""}${value.toFixed(3)}`;
}

/** Whole and half steps across the drawn range, always including zero. */
function tickValues(floor: number): number[] {
  const ticks: number[] = [];
  for (let value = Math.ceil(floor * 2) / 2; value <= CEILING; value += 0.5) {
    ticks.push(Number(value.toFixed(1)));
  }
  return ticks;
}

/**
 * The marks for a holdout result: the two regimes, plus the home score as the
 * reference the transfer is being judged against.
 */
export function marksFor(
  byEvidence: TransferScore[],
  homeR2: number | undefined,
): { marks: Mark[]; floor: number } {
  const inside = byEvidence.find((row) => row.label.startsWith("inside"));
  const outside = byEvidence.find((row) => row.label.startsWith("outside"));

  const marks: Mark[] = [];
  if (inside) {
    marks.push({
      label: "inside its training ranges",
      r2: inside.r2,
      colour: "var(--gauge-deep)",
      level: 0,
    });
  }
  if (outside) {
    marks.push({
      label: "extrapolating",
      r2: outside.r2,
      colour: "var(--flag)",
      level: 0,
    });
  }
  if (typeof homeR2 === "number") {
    // Drawn as a datum rather than a needle: it lands within 0.06 of the
    // in-range mark, which is the finding, and as a needle its leader line ran
    // straight through that mark's own label.
    marks.push({
      label: "the same model at home",
      r2: homeR2,
      colour: "var(--ink-3)",
      reference: true,
      level: 0,
    });
  }

  const lowest = Math.min(...marks.map((mark) => mark.r2), 0);
  return { marks, floor: Math.min(-1, Math.floor(lowest * 10) / 10 - 0.1) };
}
