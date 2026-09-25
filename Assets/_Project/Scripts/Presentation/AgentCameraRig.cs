using UnityEngine;

namespace JevNpcBrain.Presentation
{
    using Core;

    /// <summary>
    /// Builds the two per-agent shots: a helmet cam and an over-the-shoulder.
    ///
    /// The helmet cam rides the head bone, so it inherits the animation -- the bob
    /// on a run, the settle when the agent stops, the flinch on a hit. Raw bone
    /// motion is too much, though: the aim clips press the cheek onto the stock
    /// and tip the head sideways, and a lens bolted to that is unwatchable. So the
    /// shot keeps part of the head's rotation and takes the rest from where the
    /// agent is facing, like a gimbal. When the agent dies the gimbal lets go and
    /// the camera goes down with the body.
    ///
    /// The shoulder cam trails with a little lag and pulls in when a wall or cover
    /// gets between it and the agent -- the spawns sit close enough to the
    /// perimeter that a rigid offset would start every round inside a wall.
    ///
    /// Both cameras live at the scene root and are driven in world space.
    /// Parented to the agent they would inherit its 540-degrees-a-second turns
    /// instantly and no smoothing could touch them.
    ///
    /// Runs after AgentVisual, which turns the rig into its aim stance in
    /// LateUpdate; reading the head any earlier would lag a frame behind the body.
    /// </summary>
    [DefaultExecutionOrder(100)]
    public sealed class AgentCameraRig : MonoBehaviour
    {
        [Header("Helmet cam")]
        [Tooltip("Mount point relative to the head bone, in the character's frame: " +
                 "x right, y up, z forward. Ahead of the goggles, so the helmet " +
                 "never fills the lens.")]
        public Vector3 HelmetMount = new Vector3(0f, 0.1f, 0.17f);

        public float HelmetFieldOfView = 80f;
        public float HelmetNearClip = 0.05f;

        [Tooltip("How much of the head's own rotation the shot keeps. 0 is a gimbal " +
                 "locked to the agent's facing, 1 is the raw bone.")]
        [Range(0f, 1f)] public float HeadMotion = 0.35f;

        public float HelmetSmoothing = 14f;

        [Tooltip("Pitch limits while alive, in degrees. The sprint and reload clips " +
                 "drop the head far enough to fill the lens with boots.")]
        public float MaxLookDown = 20f;
        public float MaxLookUp = 15f;

        [Header("Over the shoulder")]
        public Vector3 ShoulderOffset = new Vector3(0.62f, 1.62f, -2.3f);
        public float ShoulderFieldOfView = 58f;
        public float ShoulderPitch = 6f;
        public float ShoulderSmoothing = 8f;

        [Tooltip("Gap kept between the lens and whatever it would otherwise clip into.")]
        public float ShoulderClearance = 0.25f;

        /// <summary>A jump bigger than this in one frame is a respawn, not motion.</summary>
        private const float TeleportMetres = 2f;

        public Camera Helmet { get; private set; }
        public Camera OverShoulder { get; private set; }
        public NpcAgent Agent { get; private set; }

        private Transform _head;
        private Quaternion _headToFacing = Quaternion.identity;
        private Vector3 _lastPosition;
        private bool _snap = true;

        public void Build(NpcAgent agent, Animator animator)
        {
            Agent = agent;

            _head = animator != null && animator.isHuman
                ? animator.GetBoneTransform(HumanBodyBones.Head)
                : null;

            // Head bones carry arbitrary orientations, the same problem as the
            // weapon. Measure the bone against the character's own facing once,
            // in the bind pose, and carry that offset from then on.
            if (_head != null) _headToFacing = Quaternion.Inverse(_head.rotation) * transform.rotation;

            Helmet = MakeCamera("Helmet", HelmetFieldOfView, HelmetNearClip);
            OverShoulder = MakeCamera("Shoulder", ShoulderFieldOfView, 0.15f);
            _lastPosition = transform.position;
        }

        private void LateUpdate()
        {
            if (Agent == null) return;

            if ((transform.position - _lastPosition).sqrMagnitude > TeleportMetres * TeleportMetres) _snap = true;
            _lastPosition = transform.position;

            float dt = Time.deltaTime;
            UpdateHelmet(dt);
            UpdateShoulder(dt);
            _snap = false;
        }

        private void UpdateHelmet(float dt)
        {
            if (Helmet == null) return;

            var facing = Quaternion.LookRotation(transform.forward, Vector3.up);
            Vector3 position;
            Quaternion look;

            if (_head != null)
            {
                // The mount rides the stabilised frame, not the raw bone: on the aim
                // clips the head drops onto the stock, and a lens that followed it
                // down ended up sitting on the buffer tube.
                var headLook = _head.rotation * _headToFacing;
                look = Agent.IsAlive
                    ? ClampPitch(Quaternion.Slerp(facing, headLook, HeadMotion), MaxLookDown, MaxLookUp)
                    : headLook;
                position = _head.position + look * HelmetMount;
            }
            else
            {
                position = Agent.EyePosition;
                look = facing;
            }

            var cam = Helmet.transform;
            cam.SetPositionAndRotation(position,
                _snap ? look : Quaternion.Slerp(cam.rotation, look, 1f - Mathf.Exp(-HelmetSmoothing * dt)));
        }

        /// <summary>Tilts a rotation about its own right axis until its pitch is inside the limits.</summary>
        private static Quaternion ClampPitch(Quaternion rotation, float down, float up)
        {
            var forward = rotation * Vector3.forward;
            float pitch = Mathf.Asin(Mathf.Clamp(forward.y, -1f, 1f)) * Mathf.Rad2Deg;
            float clamped = Mathf.Clamp(pitch, -down, up);
            if (Mathf.Approximately(pitch, clamped)) return rotation;

            // A positive turn about +X tips the forward axis down, hence the sign.
            return Quaternion.AngleAxis(pitch - clamped, rotation * Vector3.right) * rotation;
        }

        private void UpdateShoulder(float dt)
        {
            if (OverShoulder == null) return;

            var pivot = transform.TransformPoint(new Vector3(ShoulderOffset.x * 0.5f, ShoulderOffset.y, 0f));
            var desired = transform.TransformPoint(ShoulderOffset);

            var reach = desired - pivot;
            float length = reach.magnitude;

            if (length > 0.01f && Physics.SphereCast(pivot, 0.2f, reach / length, out var hit, length,
                    Physics.DefaultRaycastLayers, QueryTriggerInteraction.Ignore))
                desired = pivot + reach / length * Mathf.Max(hit.distance - ShoulderClearance, 0.3f);

            var look = transform.rotation * Quaternion.Euler(ShoulderPitch, 0f, 0f);
            float k = _snap ? 1f : 1f - Mathf.Exp(-ShoulderSmoothing * dt);

            var cam = OverShoulder.transform;
            cam.SetPositionAndRotation(Vector3.Lerp(cam.position, desired, k), Quaternion.Slerp(cam.rotation, look, k));
        }

        /// <summary>
        /// Off air until the director cuts to it. No AudioListener: Unity allows
        /// exactly one, and it stays on the spectator camera so a cut never kills
        /// the audio.
        /// </summary>
        private Camera MakeCamera(string label, float fov, float nearClip)
        {
            var go = new GameObject("Cam_" + label + " (" + Agent.DisplayName + ")");
            var cam = go.AddComponent<Camera>();
            cam.fieldOfView = fov;
            cam.nearClipPlane = nearClip;
            cam.farClipPlane = 200f;
            cam.enabled = false;
            return cam;
        }

        private void OnDestroy()
        {
            if (Helmet != null) Destroy(Helmet.gameObject);
            if (OverShoulder != null) Destroy(OverShoulder.gameObject);
        }
    }
}
