using UnityEngine;

namespace JevNpcBrain.Presentation
{
    /// <summary>
    /// How a rifle model sits in the hands. Lives on the weapon prefab, so swapping
    /// rifles is swapping prefabs and the fit travels with the asset -- instead of
    /// living in code defaults on a component that only exists at runtime, where
    /// no Inspector tweak could ever stick.
    ///
    /// Convention: the prefab root is the pistol grip, +Z runs down the barrel and
    /// +Y is up. The vendor model is a child, rotated and scaled into that frame,
    /// so a store asset never has to be edited.
    /// </summary>
    public sealed class WeaponModel : MonoBehaviour
    {
        [Tooltip("The root's pose in the RightHand bone's local space. Fitted in " +
                 "the editor against the aim and fire clips. To retune: enter play " +
                 "mode, select the rifle under the hand bone, move it until it sits " +
                 "right, and copy its local position and rotation back here.")]
        public Vector3 HandPosition;

        public Vector3 HandEuler;

        [Tooltip("Where the off-hand wrist is pinned, under the handguard.")]
        public Transform LeftGrip;

        [Tooltip("Barrel tip, for muzzle effects. Damage still fires from the eye, " +
                 "identically for both teams.")]
        public Transform Muzzle;
    }
}
