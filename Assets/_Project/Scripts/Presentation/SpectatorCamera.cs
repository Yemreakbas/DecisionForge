using UnityEngine;

namespace JevNpcBrain.Presentation
{
    using Arena;
    using Core;

    /// <summary>
    /// Top-down match camera. Frames the whole arena by default and can tighten
    /// onto the live action, which is what makes a captain duel watchable without
    /// anyone driving the camera by hand.
    /// </summary>
    public sealed class SpectatorCamera : MonoBehaviour
    {
        public float Height = 42f;
        public float Tilt = 68f;

        [Tooltip("0 = always frame the whole arena, 1 = always follow the fight.")]
        [Range(0f, 1f)] public float FollowAction = 0.55f;

        public float Damping = 2.5f;

        private Vector3 _focus;

        private void Start()
        {
            var layout = ArenaLayout.Current;
            _focus = layout != null ? layout.CenterPosition : Vector3.zero;
        }

        private void LateUpdate()
        {
            var layout = ArenaLayout.Current;
            if (layout == null) return;

            _focus = Vector3.Lerp(_focus,
                Vector3.Lerp(layout.CenterPosition, ActionCentroid(layout), FollowAction),
                Time.deltaTime * Damping);

            var offset = Quaternion.Euler(Tilt, 0f, 0f) * Vector3.back * Height;
            transform.position = _focus + offset;
            transform.rotation = Quaternion.LookRotation(_focus - transform.position);
        }

        /// <summary>Midpoint of everyone still standing.</summary>
        private Vector3 ActionCentroid(ArenaLayout layout)
        {
            var sum = Vector3.zero;
            int count = 0;

            for (int i = 0; i < NpcAgent.All.Count; i++)
            {
                var agent = NpcAgent.All[i];
                if (!agent.IsAlive) continue;
                sum += agent.transform.position;
                count++;
            }

            return count > 0 ? sum / count : layout.CenterPosition;
        }
    }
}
