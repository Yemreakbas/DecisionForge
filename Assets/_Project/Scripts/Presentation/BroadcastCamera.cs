using UnityEngine;

namespace JevNpcBrain.Presentation
{
    using Arena;
    using Core;

    /// <summary>
    /// One broadcast camera in the arena. Three rigs, the three shots a sports
    /// director reaches for:
    ///
    ///   Track -- fixed mount that pans and zooms to keep the whole fight in
    ///            frame. The towers and the wide shot.
    ///   Orbit -- a slow circle around the arena. Openers and breathers.
    ///   Duel  -- the two-shot: side-on to the line between the two captains and
    ///            far enough back to hold both. The one shot that shows a duel as
    ///            a duel -- who is where, and who turned first.
    ///
    /// Every rig keeps updating while it is off air (the director disables the
    /// Camera component, not the GameObject), so a cut never lands on a stale
    /// frame and then swings into place.
    /// </summary>
    [DefaultExecutionOrder(150)]
    [RequireComponent(typeof(Camera))]
    public sealed class BroadcastCamera : MonoBehaviour
    {
        public enum Rig { Track, Orbit, Duel }

        public Rig Mode = Rig.Track;

        [Header("Lens")]
        public float MinFieldOfView = 22f;
        public float MaxFieldOfView = 60f;

        [Tooltip("Metres of room kept around the fighters.")]
        public float Margin = 3.5f;

        [Tooltip("How quickly the shot settles on the action. Higher is snappier.")]
        public float Damping = 2.5f;

        [Header("Track")]
        [Tooltip("Follow the nearest fighter on a long lens instead of framing the " +
                 "whole fight. The towers do this: framing both captains from a " +
                 "corner is just a worse wide shot, while four iso cams that each " +
                 "own a corner give the director a close-up anywhere in the arena.")]
        public bool FollowNearest;

        [Tooltip("Metres of room kept around a followed fighter.")]
        public float FollowRadius = 2.5f;

        [Tooltip("How much closer another fighter must be before the lens switches to them.")]
        public float SwitchMetres = 4f;

        [Header("Orbit")]
        public float OrbitRadius = 17f;
        public float OrbitHeight = 9f;
        public float OrbitDegreesPerSecond = 6f;

        [Header("Duel")]
        public float DuelHeight = 3.2f;
        public float DuelMinDistance = 7f;

        [Tooltip("Distance from the pair as a fraction of their separation.")]
        public float DuelDistanceScale = 0.8f;

        [Tooltip("Metres the camera keeps in from the arena radius, so the " +
                 "perimeter wall never ends up between the lens and the fight.")]
        public float WallClearance = 3f;

        private Camera _camera;
        private Vector3 _position;
        private Vector3 _lookAt;
        private float _orbitDegrees;
        private float _duelSide = 1f;
        private Vector3 _lastMount;
        private bool _settled;

        /// <summary>The fighter a FollowNearest camera is on. Null for every other rig.</summary>
        public NpcAgent Subject { get; private set; }

        /// <summary>False when cover stands between the lens and the subject.</summary>
        public bool SubjectInView { get; private set; }

        private void LateUpdate()
        {
            var layout = ArenaLayout.Current;
            if (layout == null) return;

            if (_camera == null) _camera = GetComponent<Camera>();

            // The first frame snaps; after that everything eases.
            float k = _settled ? 1f - Mathf.Exp(-Damping * Time.deltaTime) : 1f;

            switch (Mode)
            {
                case Rig.Orbit: TickOrbit(layout, k); break;
                case Rig.Duel: TickDuel(layout, k); break;
                default: TickTrack(k); break;
            }

            _settled = true;
        }

        /// <summary>Fixed mount: aim at the action, or at one fighter, and zoom until it fits.</summary>
        private void TickTrack(float k)
        {
            Vector3 focus;
            float radius;

            if (FollowNearest && UpdateSubject())
            {
                focus = Subject.transform.position;
                radius = FollowRadius;
            }
            else
            {
                ActionFocus.Measure(out focus, out float spread);
                radius = spread + Margin;
            }

            _lookAt = Vector3.Lerp(_lookAt, focus + Vector3.up * 1.1f, k);

            var toTarget = _lookAt - transform.position;
            if (toTarget.sqrMagnitude > 0.01f) transform.rotation = Quaternion.LookRotation(toTarget);

            Zoom(radius, toTarget.magnitude, k);
        }

        /// <summary>
        /// Nearest fighter still in the action, held until another is clearly
        /// closer -- two fighters at similar range would otherwise flick the lens
        /// back and forth between them.
        /// </summary>
        private bool UpdateSubject()
        {
            var all = NpcAgent.All;
            NpcAgent best = null;
            float bestDistance = float.MaxValue;
            float currentDistance = float.MaxValue;

            for (int i = 0; i < all.Count; i++)
            {
                var agent = all[i];
                if (!ActionFocus.InAction(agent)) continue;

                float distance = Vector3.Distance(transform.position, agent.transform.position);
                if (agent == Subject) currentDistance = distance;

                if (distance >= bestDistance) continue;
                bestDistance = distance;
                best = agent;
            }

            if (best == null)
            {
                Subject = null;
                SubjectInView = false;
                return false;
            }

            if (currentDistance > bestDistance + SwitchMetres) Subject = best;

            SubjectInView = CanSee(Subject);
            return true;
        }

        private bool CanSee(NpcAgent agent)
        {
            var target = agent.transform.position + Vector3.up * 1.3f;

            if (!Physics.Linecast(transform.position, target, out var hit,
                    Physics.DefaultRaycastLayers, QueryTriggerInteraction.Ignore))
                return true;

            return hit.collider.GetComponentInParent<NpcAgent>() == agent;
        }

        /// <summary>Slow circle, watching a point halfway between the centre and the fight.</summary>
        private void TickOrbit(ArenaLayout layout, float k)
        {
            var center = layout.CenterPosition;

            if (!_settled)
            {
                var start = transform.position - center;
                _orbitDegrees = Mathf.Atan2(start.x, start.z) * Mathf.Rad2Deg;
            }

            _orbitDegrees += OrbitDegreesPerSecond * Time.deltaTime;

            var position = center
                           + Quaternion.Euler(0f, _orbitDegrees, 0f) * Vector3.forward * OrbitRadius
                           + Vector3.up * OrbitHeight;

            ActionFocus.Measure(out var focus, out float spread);
            _lookAt = Vector3.Lerp(_lookAt, Vector3.Lerp(center, focus, 0.5f) + Vector3.up, k);

            transform.SetPositionAndRotation(position, Quaternion.LookRotation(_lookAt - position));
            Zoom(Mathf.Max(spread + Margin, layout.Radius * 0.6f), Vector3.Distance(position, _lookAt), k);
        }

        /// <summary>
        /// The two-shot. Side-on to the line between the captains, and it stays on
        /// its own side of that line: crossing it swaps who is on the left of the
        /// screen, which reads as the fighters trading places. It only goes across
        /// when the wall leaves no room on its side -- and then it cuts, rather
        /// than swooping through the middle of the fight.
        /// </summary>
        private void TickDuel(ArenaLayout layout, float k)
        {
            if (!FindCaptains(out var a, out var b))
            {
                TickTrack(k);
                return;
            }

            var pa = a.transform.position;
            var pb = b.transform.position;
            var mid = (pa + pb) * 0.5f;

            var line = Flat(pb - pa);
            float separation = line.magnitude;
            var along = separation > 0.01f ? line / separation : Vector3.forward;
            var side = new Vector3(-along.z, 0f, along.x);

            float distance = Mathf.Max(DuelMinDistance, separation * DuelDistanceScale + 3f);

            var ours = Mount(layout, mid, side * _duelSide, distance);
            var theirs = Mount(layout, mid, -side * _duelSide, distance);

            if (Flat(theirs - mid).magnitude > Flat(ours - mid).magnitude + 4f)
            {
                _duelSide = -_duelSide;
                ours = theirs;
            }

            // Cut rather than glide whenever the mount jumps: a side flip, or a new
            // round teleporting both captains. The midpoint alone misses the
            // respawn -- it barely moves, while the separation goes from a couple
            // of metres to the full width of the arena.
            bool cut = !_settled || (ours - _lastMount).sqrMagnitude > 9f;
            _lastMount = ours;
            if (cut) k = 1f;

            _position = Vector3.Lerp(_position, ours, k);
            _lookAt = Vector3.Lerp(_lookAt, mid + Vector3.up * 1.2f, k);
            transform.SetPositionAndRotation(_position, Quaternion.LookRotation(_lookAt - _position));

            Zoom(separation * 0.5f + Margin, Vector3.Distance(_position, _lookAt), k);
        }

        private Vector3 Mount(ArenaLayout layout, Vector3 mid, Vector3 direction, float distance)
        {
            var local = Flat(mid + direction * distance - layout.CenterPosition);
            float limit = Mathf.Max(layout.Radius - WallClearance, 1f);
            if (local.magnitude > limit) local = local.normalized * limit;
            return layout.CenterPosition + local + Vector3.up * DuelHeight;
        }

        /// <summary>
        /// Narrowest lens that still holds a circle of this radius across the
        /// screen. Unity's field of view is vertical and a fight spreads
        /// sideways, hence the aspect. Never so tight that a standing agent is
        /// cropped.
        /// </summary>
        private void Zoom(float radius, float distance, float k)
        {
            if (_camera == null || distance < 0.1f) return;

            float aspect = Mathf.Max(_camera.aspect, 0.1f);
            float across = 2f * Mathf.Atan(radius / (distance * aspect)) * Mathf.Rad2Deg;
            float standing = 2f * Mathf.Atan(1.6f / distance) * Mathf.Rad2Deg;

            float target = Mathf.Clamp(Mathf.Max(across, standing), MinFieldOfView, MaxFieldOfView);
            _camera.fieldOfView = Mathf.Lerp(_camera.fieldOfView, target, k);
        }

        private static bool FindCaptains(out NpcAgent a, out NpcAgent b)
        {
            a = b = null;
            var all = NpcAgent.All;

            for (int i = 0; i < all.Count; i++)
            {
                var agent = all[i];
                if (!agent.IsCaptain) continue;
                if (agent.Team == 0) a = agent;
                else if (agent.Team == 1) b = agent;
            }

            return a != null && b != null;
        }

        private static Vector3 Flat(Vector3 v) => new Vector3(v.x, 0f, v.z);
    }
}
