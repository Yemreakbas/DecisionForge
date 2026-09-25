using UnityEngine;

namespace JevNpcBrain.Core
{
    /// <summary>
    /// The non-spatial half of an observation: everything about the agent and the
    /// match that does not belong on a grid. Fixed length and fixed order, because
    /// this array is fed straight into an ONNX model -- reordering a field silently
    /// invalidates every trained checkpoint. Append only, never insert.
    /// All values are normalised to roughly [-1, 1].
    /// </summary>
    public struct SelfState
    {
        public const int Length = 24;

        /// <summary>
        /// Field names in WriteTo order, written into every dataset's run.json so the
        /// trainer reads features by name. Append here whenever a field is appended
        /// below -- the dataset recorder refuses to start if the counts disagree.
        /// </summary>
        public static readonly string[] FieldNames =
        {
            "Health01", "Ammo01", "IsReloading", "ReloadProgress01", "Stance", "Speed01",
            "FacingSin", "FacingCos", "TimeSinceEnemySeen01", "TimeSinceDamaged01",
            "HasLineOfSight", "VisibleEnemies01", "AlliesAlive01", "AllyAverageHealth01",
            "TeamScore01", "EnemyScore01", "MatchTimeRemaining01", "InCover01",
            "NearestKnownEnemyDistance01", "Suppression01", "LastIntent01",
            "IntentHoldTime01", "DistanceToCenter01", "InCenterZone"
        };

        public float Health01;
        public float Ammo01;
        public float IsReloading;
        public float ReloadProgress01;
        public float Stance;                  // 0 = standing, 1 = crouched
        public float Speed01;
        public float FacingSin;
        public float FacingCos;
        public float TimeSinceEnemySeen01;    // saturates at MemoryHorizon
        public float TimeSinceDamaged01;
        public float HasLineOfSight;
        public float VisibleEnemies01;
        public float AlliesAlive01;
        public float AllyAverageHealth01;
        public float TeamScore01;
        public float EnemyScore01;
        public float MatchTimeRemaining01;
        public float InCover01;
        public float NearestKnownEnemyDistance01;
        public float Suppression01;
        public float LastIntent01;            // previous intent index / (Count - 1)
        public float IntentHoldTime01;        // how long we have committed to it
        public float DistanceToCenter01;
        public float InCenterZone;

        public void WriteTo(float[] destination, int offset = 0)
        {
            destination[offset + 0] = Health01;
            destination[offset + 1] = Ammo01;
            destination[offset + 2] = IsReloading;
            destination[offset + 3] = ReloadProgress01;
            destination[offset + 4] = Stance;
            destination[offset + 5] = Speed01;
            destination[offset + 6] = FacingSin;
            destination[offset + 7] = FacingCos;
            destination[offset + 8] = TimeSinceEnemySeen01;
            destination[offset + 9] = TimeSinceDamaged01;
            destination[offset + 10] = HasLineOfSight;
            destination[offset + 11] = VisibleEnemies01;
            destination[offset + 12] = AlliesAlive01;
            destination[offset + 13] = AllyAverageHealth01;
            destination[offset + 14] = TeamScore01;
            destination[offset + 15] = EnemyScore01;
            destination[offset + 16] = MatchTimeRemaining01;
            destination[offset + 17] = InCover01;
            destination[offset + 18] = NearestKnownEnemyDistance01;
            destination[offset + 19] = Suppression01;
            destination[offset + 20] = LastIntent01;
            destination[offset + 21] = IntentHoldTime01;
            destination[offset + 22] = DistanceToCenter01;
            destination[offset + 23] = InCenterZone;
        }

        public void SetFacing(Vector3 forward)
        {
            var flat = new Vector2(forward.x, forward.z).normalized;
            FacingSin = flat.x;
            FacingCos = flat.y;
        }
    }
}
