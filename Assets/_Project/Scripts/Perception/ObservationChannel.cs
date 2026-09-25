namespace JevNpcBrain.Perception
{
    /// <summary>
    /// Channels of the egocentric tactical grid. Order is part of the model
    /// contract -- see <see cref="TacticalObservation"/>.
    /// </summary>
    public enum ObservationChannel
    {
        /// <summary>0 = open floor, 0.5 = crouch-height cover, 1 = full-height blocker.</summary>
        ObstacleHeight = 0,

        /// <summary>How well this cell shelters us from currently believed threat directions.</summary>
        CoverQuality = 1,

        /// <summary>Decayed belief that an enemy occupies this cell. NOT ground truth.</summary>
        EnemyBelief = 2,

        /// <summary>Known ally occupancy.</summary>
        AllyPresence = 3,

        /// <summary>Recent gunfire and footstep energy, decayed over time.</summary>
        NoiseHeat = 4,

        /// <summary>1 if this cell is currently within our own line of sight.</summary>
        Visibility = 5,

        /// <summary>How many believed enemy positions can see this cell, normalised.</summary>
        ThreatExposure = 6
    }
}
