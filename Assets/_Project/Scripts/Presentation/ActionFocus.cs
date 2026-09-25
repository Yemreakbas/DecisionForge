using UnityEngine;

namespace JevNpcBrain.Presentation
{
    using Arena;
    using Core;

    /// <summary>
    /// Where the fight is: the midpoint of everyone still in it, and how far they
    /// spread from that point. Every camera that frames the action asks here, so
    /// they all agree on what "the action" means.
    ///
    /// A body keeps counting for a few seconds after it drops. Without that, the
    /// moment someone dies every tracking shot would lurch toward the survivor and
    /// frame the kill out of the picture -- the one moment the viewer came for.
    /// </summary>
    public static class ActionFocus
    {
        public const float LingerSeconds = 3f;

        public static bool Measure(out Vector3 centroid, out float spread)
        {
            var all = NpcAgent.All;
            var sum = Vector3.zero;
            int count = 0;

            for (int i = 0; i < all.Count; i++)
            {
                if (!InAction(all[i])) continue;
                sum += all[i].transform.position;
                count++;
            }

            spread = 0f;

            if (count == 0)
            {
                var layout = ArenaLayout.Current;
                centroid = layout != null ? layout.CenterPosition : Vector3.zero;
                return false;
            }

            centroid = sum / count;

            for (int i = 0; i < all.Count; i++)
            {
                if (!InAction(all[i])) continue;
                var d = all[i].transform.position - centroid;
                d.y = 0f;
                spread = Mathf.Max(spread, d.magnitude);
            }

            return true;
        }

        public static bool InAction(NpcAgent agent)
        {
            if (agent == null) return false;
            if (agent.IsAlive) return true;
            return agent.Health != null && Time.time - agent.Health.LastDamagedAt < LingerSeconds;
        }
    }
}
