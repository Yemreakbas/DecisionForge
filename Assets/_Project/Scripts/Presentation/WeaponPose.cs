using UnityEngine;

namespace JevNpcBrain.Presentation
{
    /// <summary>
    /// Keeps the off hand on the rifle's foregrip, through clips that know nothing
    /// about a rifle.
    ///
    /// Mixamo has no rifle variants of jump, vault, slide or dive. Rather than hunt
    /// for clips that do not exist, we let the empty-handed clip drive the body and
    /// pin the left hand to the weapon with Humanoid IK. The rifle itself rides the
    /// right hand bone, so it never leaves the character no matter what is playing.
    ///
    /// Lowering <see cref="GripWeight"/> releases the off hand -- which is exactly
    /// what a real operator does going over an obstacle, one hand on the rifle and
    /// one on the wall. The weakness of the borrowed clip becomes the correct
    /// behaviour.
    ///
    /// Requires "IK Pass" ticked on the Animator layer that drives the body.
    /// </summary>
    [RequireComponent(typeof(Animator))]
    public sealed class WeaponPose : MonoBehaviour
    {
        [Tooltip("Foregrip transform on the weapon model. The left hand is pinned here.")]
        public Transform LeftGrip;

        [Range(0f, 1f)] public float GripWeight = 1f;

        [Tooltip("How much of the foregrip's rotation the off hand takes, on top of " +
                 "its position. Zero keeps the clip's own wrist angle, which the " +
                 "rifle clips already get right; a marker rotation that is even " +
                 "slightly off twists the wrist visibly.")]
        [Range(0f, 1f)] public float GripRotationWeight;

        [Tooltip("Where the head looks. Left null to leave head IK alone.")]
        public Transform LookTarget;

        [Range(0f, 1f)] public float LookWeight;

        private Animator _animator;

        private void Awake() => _animator = GetComponent<Animator>();

        private void OnAnimatorIK(int layerIndex)
        {
            if (_animator == null) return;

            if (LeftGrip != null && GripWeight > 0.001f)
            {
                _animator.SetIKPositionWeight(AvatarIKGoal.LeftHand, GripWeight);
                _animator.SetIKRotationWeight(AvatarIKGoal.LeftHand, GripWeight * GripRotationWeight);
                _animator.SetIKPosition(AvatarIKGoal.LeftHand, LeftGrip.position);
                _animator.SetIKRotation(AvatarIKGoal.LeftHand, LeftGrip.rotation);
            }
            else
            {
                _animator.SetIKPositionWeight(AvatarIKGoal.LeftHand, 0f);
                _animator.SetIKRotationWeight(AvatarIKGoal.LeftHand, 0f);
            }

            if (LookTarget == null || LookWeight <= 0.001f)
            {
                _animator.SetLookAtWeight(0f);
                return;
            }

            _animator.SetLookAtWeight(LookWeight, 0.25f, 0.6f);
            _animator.SetLookAtPosition(LookTarget.position);
        }
    }
}
