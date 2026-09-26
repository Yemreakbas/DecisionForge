using System.Collections.Generic;
using UnityEngine;

namespace JevNpcBrain.Tactical
{
    using Core;
    using Perception;
    using Safety;

    /// <summary>
    /// The experimental brain: a planner over a learned world model.
    ///
    /// Every tick it encodes the observation into a latent, imagines each of the
    /// ten intents held for the next 1.5 s (<see cref="JevRuntime.Imagine"/>), reads
    /// the probes at every imagined step, and picks the future it likes best. It
    /// never reads an enemy transform or anything else the baseline cannot see:
    /// the observation in is the same instance UtilityBrain gets, the intent out is
    /// from the same set, and the reflex layer, weapon and tick rate are shared.
    ///
    /// What it likes is <see cref="Weights"/>: imminent death and damage are bad,
    /// exposure is bad, cover is good, a won round is good, the centre slightly so.
    /// Nobody wrote down *how* to get there -- which manoeuvre leads to cover or
    /// out of a crossfire is the world model's prediction, not a rule. That is the
    /// difference under test.
    ///
    /// The position probes are display-only: they turn each imagined future into a
    /// path for <see cref="Presentation.ImaginationTrails"/> and never enter a score.
    /// </summary>
    public sealed class JevBrain : ITacticalBrain
    {
        /// <summary>
        /// Scoring. Mirrors <c>WEIGHTS</c> in <c>Training/plan_offline.py</c>, which
        /// checks a weight set offline before a match is spent on it.
        /// </summary>
        [System.Serializable]
        public sealed class Weights
        {
            public float DeathSoon = -1.5f;
            public float HurtSoon = -0.75f;
            public float Exposure = -0.25f;
            public float InCover = 0.25f;
            public float RoundWon = 1.0f;
            public float InCenter = 0.1f;

            [Tooltip("Per-tick discount over the imagined steps: the near future is predicted better.")]
            public float Discount = 0.93f;

            // Same hysteresis mechanism as UtilityBrain, scaled to this score range
            // (the median gap between the two best candidates is ~0.06).
            public float CommitmentBonus = 0.08f;
            public float MinCommitSeconds = 0.6f;

            public static Weights Captain() => new Weights { CommitmentBonus = 0.1f };
        }

        public string Label => "JEV";
        public IReadOnlyList<ScoredIntent> LastScores => _scores;

        /// <summary>Imagined path of every candidate, world space, anchored where the agent stood when it decided.</summary>
        public readonly Vector3[,] Rollouts;

        /// <summary>Candidate (= intent index) the last decision executed, or -1.</summary>
        public int Chosen { get; private set; } = -1;

        public bool HasRollouts { get; private set; }
        public int Horizon => _rt.Horizon;
        public int Candidates => _rt.Candidates;

        private readonly JevRuntime _rt;
        private readonly Weights _w;
        private readonly List<ScoredIntent> _scores = new List<ScoredIntent>(TacticalIntents.Count);
        private readonly float[] _stepWeight;
        private readonly int _death, _hurt, _exposure, _cover, _won, _center, _posX, _posZ;
        private readonly float _metresPerUnit;

        private TacticalIntent _current = TacticalIntent.Hold;
        private float _committedAt = float.NegativeInfinity;

        public JevBrain(JevRuntime runtime, Weights weights = null)
        {
            _rt = runtime;
            _w = weights ?? new Weights();

            var c = runtime.Contract;
            _death = c.ProbeIndex("death_soon");
            _hurt = c.ProbeIndex("hurt_soon");
            _exposure = c.ProbeIndex("exposure");
            _cover = c.ProbeIndex("in_cover");
            _won = c.ProbeIndex("round_won");
            _center = c.ProbeIndex("in_center");
            _posX = c.ProbeIndex(c.position_probes.x);
            _posZ = c.ProbeIndex(c.position_probes.z);
            _metresPerUnit = 2f * c.position_probes.half_extent;

            _stepWeight = new float[runtime.Horizon];
            float total = 0f;
            for (int k = 0; k < _stepWeight.Length; k++) total += _stepWeight[k] = Mathf.Pow(_w.Discount, k);
            for (int k = 0; k < _stepWeight.Length; k++) _stepWeight[k] /= total;

            Rollouts = new Vector3[runtime.Candidates, runtime.Horizon];
        }

        public void Reset()
        {
            _current = TacticalIntent.Hold;
            _committedAt = float.NegativeInfinity;
            _scores.Clear();
            HasRollouts = false;
            Chosen = -1;
        }

        public TacticalIntent Decide(TacticalObservation obs)
        {
            _scores.Clear();
            _rt.Imagine(obs);

            var p = _rt.Probes;
            int horizon = _rt.Horizon, probes = _rt.ProbeCount;
            var best = TacticalIntent.Hold;
            float bestScore = float.NegativeInfinity;

            for (int c = 0; c < _rt.Candidates; c++)
            {
                var intent = TacticalIntents.All[c];
                float score = 0f;
                int first = c * horizon * probes;

                for (int k = 0; k < horizon; k++)
                {
                    int o = first + k * probes;
                    score += _stepWeight[k] * (
                        _w.DeathSoon * p[o + _death] + _w.HurtSoon * p[o + _hurt]
                        + _w.Exposure * p[o + _exposure] + _w.InCover * p[o + _cover]
                        + _w.RoundWon * p[o + _won] + _w.InCenter * p[o + _center]);

                    // Displacement from the first imagined step, so the absolute
                    // bias of the position probe (~1 m) drops out of the trail.
                    Rollouts[c, k] = obs.Origin + new Vector3(
                        (p[o + _posX] - p[first + _posX]) * _metresPerUnit, 0f,
                        (p[o + _posZ] - p[first + _posZ]) * _metresPerUnit);
                }

                string veto = ConstraintFilter.Veto(intent, obs);
                _scores.Add(new ScoredIntent(intent, score, veto));
                if (veto != null) continue;

                float committed = score + (intent == _current ? _w.CommitmentBonus : 0f);
                if (committed <= bestScore) continue;
                bestScore = committed;
                best = intent;
            }

            // Hysteresis, as in UtilityBrain -- unless the safety layer has since
            // ruled the current intent out, which ends any commitment.
            bool currentAllowed = ConstraintFilter.IsAllowed(_current, obs);
            if (best != _current && currentAllowed && obs.Timestamp - _committedAt < _w.MinCommitSeconds)
                best = _current;

            if (best != _current)
            {
                _current = best;
                _committedAt = obs.Timestamp;
            }

            Chosen = (int)_current;
            HasRollouts = true;
            return _current;
        }
    }
}
