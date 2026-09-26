using System.Collections.Generic;
using UnityEngine;

namespace JevNpcBrain.Presentation
{
    using Core;
    using Tactical;

    /// <summary>
    /// Draws what a JEV brain imagined on its last tick: one path per candidate
    /// intent, 1.5 s into the future. The chosen future is a bright line in the
    /// team colour; the rejected ones are thin trails that fade out along their
    /// length; vetoed ones are not drawn. The utility baseline has nothing to draw
    /// here -- it only has a table of numbers -- and that contrast is the point.
    ///
    /// Paths come from the model's display-only position probes, anchored where
    /// the agent stood when it decided. Presentation only: nothing reads them back.
    /// </summary>
    public sealed class ImaginationTrails : MonoBehaviour
    {
        public bool Show = true;
        public float Height = 0.12f;
        // Sized for the 40 m top-down shot, where 0.1 m is about one pixel.
        public float ChosenWidth = 0.25f;
        public float RejectedWidth = 0.09f;
        [Range(0f, 1f)] public float RejectedAlpha = 0.4f;

        private static readonly Color[] TeamColours =
        {
            new Color(0.25f, 0.55f, 0.95f),
            new Color(0.95f, 0.35f, 0.30f)
        };

        private readonly Dictionary<NpcAgent, LineRenderer[]> _lines = new Dictionary<NpcAgent, LineRenderer[]>();
        private Transform _root;
        private Material _material;
        private Vector3[] _points;

        private void Awake()
        {
            _root = new GameObject("ImaginationTrails").transform;

            // Sprites/Default blends vertex colour and alpha, which the fade needs.
            // A player build only carries shaders something references (see
            // AgentVisual.BuildTeamRing), so fall back to one already in use.
            var shader = Shader.Find("Sprites/Default");
            if (shader == null) shader = Shader.Find("Universal Render Pipeline/Unlit");
            if (shader == null)
            {
                var any = FindAnyObjectByType<Renderer>();
                if (any != null && any.sharedMaterial != null) shader = any.sharedMaterial.shader;
            }
            if (shader != null) _material = new Material(shader) { name = "ImaginationTrail" };
        }

        private void LateUpdate()
        {
            if (_material == null) return;

            foreach (var agent in NpcAgent.All)
            {
                var jev = agent.Brain as JevBrain;
                bool visible = Show && jev != null && jev.HasRollouts && agent.IsAlive;

                if (!_lines.TryGetValue(agent, out var lines))
                {
                    if (!visible) continue;
                    lines = Create(agent, jev.Candidates);
                    _lines.Add(agent, lines);
                }

                for (int c = 0; c < lines.Length; c++)
                {
                    bool vetoed = visible && c < jev.LastScores.Count && jev.LastScores[c].Reason != null;
                    lines[c].enabled = visible && !vetoed;
                    if (lines[c].enabled) Draw(lines[c], jev, c, agent.Team);
                }
            }
        }

        private LineRenderer[] Create(NpcAgent agent, int candidates)
        {
            var lines = new LineRenderer[candidates];
            for (int c = 0; c < candidates; c++)
            {
                var go = new GameObject($"{agent.name}_{TacticalIntents.All[c]}");
                go.transform.SetParent(_root, false);
                var line = go.AddComponent<LineRenderer>();
                line.sharedMaterial = _material;
                line.useWorldSpace = true;
                line.numCapVertices = 2;
                line.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
                line.receiveShadows = false;
                lines[c] = line;
            }
            return lines;
        }

        private void Draw(LineRenderer line, JevBrain jev, int candidate, int team)
        {
            int h = jev.Horizon;
            if (_points == null || _points.Length != h) _points = new Vector3[h];
            for (int k = 0; k < h; k++) _points[k] = jev.Rollouts[candidate, k] + Vector3.up * Height;
            line.positionCount = h;
            line.SetPositions(_points);

            var colour = TeamColours[Mathf.Clamp(team, 0, 1)];
            bool chosen = candidate == jev.Chosen;
            float width = chosen ? ChosenWidth : RejectedWidth;
            line.startWidth = width;
            line.endWidth = chosen ? width : width * 0.5f;

            var bright = chosen ? Color.Lerp(colour, Color.white, 0.35f) : colour;
            float alpha = chosen ? 1f : RejectedAlpha;
            line.startColor = new Color(bright.r, bright.g, bright.b, alpha);
            line.endColor = new Color(bright.r, bright.g, bright.b, chosen ? alpha : 0f);
        }

        private void OnDestroy()
        {
            if (_root != null) Destroy(_root.gameObject);
            if (_material != null) Destroy(_material);
        }
    }
}
