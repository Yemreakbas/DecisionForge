using UnityEngine;

namespace JevNpcBrain.Presentation
{
    using Core;
    using Match;
    using Tactical;

    /// <summary>
    /// Draws what each agent is thinking, live.
    ///
    /// This is not a debug view that gets stripped later -- it is the point of the
    /// content. A viewer cannot see the difference between two architectures by
    /// watching capsules shoot each other; they can see it by watching one brain's
    /// reasoning next to the other's. For the utility team that means a score
    /// table. For the JEV team in phase 4 it will mean the rejected rollouts
    /// fading out around the chosen one.
    /// </summary>
    public sealed class ThoughtOverlay : MonoBehaviour
    {
        public MatchDirector Director;
        public CameraDirector Cameras;
        public bool Show = true;
        public int TopIntents = 4;

        private GUIStyle _header;
        private GUIStyle _line;
        private Texture2D _barTexture;

        private static readonly Color TeamAColor = new Color(0.35f, 0.65f, 1f);
        private static readonly Color TeamBColor = new Color(1f, 0.45f, 0.38f);

        private void Awake()
        {
            if (Director == null) Director = FindAnyObjectByType<MatchDirector>();
            if (Cameras == null) Cameras = FindAnyObjectByType<CameraDirector>();
            _barTexture = Texture2D.whiteTexture;
        }

        /// <summary>H hides every overlay, for clean footage.</summary>
        private void Update()
        {
#if ENABLE_INPUT_SYSTEM
            var keyboard = UnityEngine.InputSystem.Keyboard.current;
            if (keyboard != null && keyboard.hKey.wasPressedThisFrame) Show = !Show;
#endif
        }

        private void EnsureStyles()
        {
            if (_header != null) return;

            _header = new GUIStyle(GUI.skin.label)
            {
                fontSize = 14,
                fontStyle = FontStyle.Bold
            };

            _line = new GUIStyle(GUI.skin.label) { fontSize = 12 };
        }

        private void OnGUI()
        {
            if (!Show) return;
            EnsureStyles();

            DrawScoreline();
            DrawCameraBar();

            float y = 96f;
            for (int i = 0; i < NpcAgent.All.Count; i++)
            {
                var agent = NpcAgent.All[i];
                float x = agent.Team == 0 ? 12f : Screen.width - 272f;
                DrawAgentPanel(agent, x, ref y, agent.Team);
            }
        }

        /// <summary>
        /// The shot list doubles as the control legend. Anyone who picks this up
        /// later -- or watches a recording -- can see what the keys do without
        /// being told.
        /// </summary>
        private void DrawCameraBar()
        {
            if (Cameras == null) return;

            GUI.Label(new Rect(12f, 52f, Screen.width - 24f, 20f),
                "CAM  " + Cameras.CurrentLabel + (Cameras.AutoDirect ? "   (auto)" : "") +
                "      1 tactical  2 duel  3 wide  4 towers  5 orbit  6/7 helmet  8/9 shoulder" +
                "   Tab next  V versus  P inset " + (Cameras.PictureInPicture ? "on" : "off") +
                "  O auto  H hide", _line);
        }

        private void DrawScoreline()
        {
            if (Director == null || Director.Metrics == null) return;

            var a = Director.Metrics.Teams[0];
            var b = Director.Metrics.Teams[1];

            GUI.Label(new Rect(12f, 10f, 600f, 22f),
                $"Round {Director.RoundIndex + 1}/{Director.Rounds}   " +
                $"{a.BrainLabel} {a.RoundsWon} - {b.RoundsWon} {b.BrainLabel}", _header);

            GUI.Label(new Rect(12f, 32f, 700f, 20f),
                $"dumb moments  {a.DumbMoments} / {b.DumbMoments}      " +
                $"exposure  {a.ExposureSeconds:F0}s / {b.ExposureSeconds:F0}s      " +
                $"decide  {a.AverageDecisionMs:F2}ms / {b.AverageDecisionMs:F2}ms", _line);
        }

        private void DrawAgentPanel(NpcAgent agent, float x, ref float y, int team)
        {
            var color = team == 0 ? TeamAColor : TeamBColor;
            float panelY = team == 0 ? y : y;

            GUI.color = color;
            GUI.Label(new Rect(x, panelY, 260f, 20f),
                $"{agent.DisplayName}{(agent.IsCaptain ? "  [C]" : "")}   " +
                $"{(agent.IsAlive ? $"hp {agent.Health.Health:F0}" : "DOWN")}", _header);

            GUI.color = Color.white;
            panelY += 20f;

            if (!agent.IsAlive || agent.Brain == null)
            {
                y = panelY + 12f;
                return;
            }

            GUI.Label(new Rect(x, panelY, 260f, 18f),
                $"{agent.Brain.Label}  ->  {TacticalIntents.Label(agent.CurrentIntent)}", _line);
            panelY += 18f;

            DrawTopScores(agent, x, ref panelY, color);

            y = panelY + 12f;
        }

        /// <summary>
        /// The top few options with their scores, so the choice is visible as a
        /// choice -- including how close the runner-up was.
        /// </summary>
        private void DrawTopScores(NpcAgent agent, float x, ref float y, Color color)
        {
            var scores = agent.Brain.LastScores;
            if (scores == null || scores.Count == 0) return;

            float min = float.MaxValue, max = float.MinValue;
            for (int i = 0; i < scores.Count; i++)
            {
                if (scores[i].Score < min) min = scores[i].Score;
                if (scores[i].Score > max) max = scores[i].Score;
            }

            float range = Mathf.Max(max - min, 0.0001f);

            for (int shown = 0; shown < TopIntents; shown++)
            {
                int best = -1;
                float bestScore = float.NegativeInfinity;

                for (int i = 0; i < scores.Count; i++)
                {
                    if (scores[i].Score <= bestScore) continue;
                    if (IsAlreadyShown(scores, i, shown, agent)) continue;
                    bestScore = scores[i].Score;
                    best = i;
                }

                if (best < 0) break;

                var entry = scores[best];
                float normalized = (entry.Score - min) / range;
                bool chosen = entry.Intent == agent.CurrentIntent;

                GUI.color = chosen ? color : new Color(1f, 1f, 1f, 0.35f);
                GUI.DrawTexture(new Rect(x, y + 3f, 120f * normalized, 10f), _barTexture);

                GUI.color = Color.white;
                GUI.Label(new Rect(x + 126f, y - 2f, 160f, 18f),
                    $"{TacticalIntents.Label(entry.Intent)} {entry.Score:F2}" +
                    (entry.Reason != null ? $"  ({entry.Reason})" : ""), _line);

                y += 16f;
                _shownMask[shown] = entry.Intent;
            }

            GUI.color = Color.white;
        }

        private readonly TacticalIntent[] _shownMask = new TacticalIntent[8];

        private bool IsAlreadyShown(System.Collections.Generic.IReadOnlyList<ScoredIntent> scores,
            int index, int shownCount, NpcAgent agent)
        {
            for (int i = 0; i < shownCount; i++)
                if (_shownMask[i] == scores[index].Intent) return true;
            return false;
        }
    }
}
