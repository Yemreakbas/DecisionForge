using System;
using System.Collections.Generic;
using Unity.InferenceEngine;
using UnityEngine;

namespace JevNpcBrain.Tactical
{
    using Core;
    using Perception;

    /// <summary>
    /// One trained world model as the game sees it: the two ONNX graphs the planner
    /// runs and the <c>model.json</c> contract written beside them by
    /// <c>Training/export_onnx.py</c>. Swapping models is swapping this asset.
    /// </summary>
    [CreateAssetMenu(menuName = "JEV/World Model", fileName = "JevModel")]
    public sealed class JevModelAsset : ScriptableObject
    {
        [Tooltip("encoder.onnx: grid (B,7,32,32) + self (B,24) -> latent (B,D)")]
        public ModelAsset Encoder;

        [Tooltip("imagine.onnx: latent (1,D) + intents (N,H,10) -> probes (N,H,P)")]
        public ModelAsset Imagine;

        [Tooltip("model.json from the same export")]
        public TextAsset Contract;

        /// <summary>Parses and validates the contract; throws on any mismatch with this build.</summary>
        public JevContract LoadContract()
        {
            if (Encoder == null || Imagine == null || Contract == null)
                throw new InvalidOperationException($"{name}: Encoder, Imagine and Contract must all be assigned.");

            var contract = JsonUtility.FromJson<JevContract>(Contract.text);
            contract.Validate(name);
            return contract;
        }
    }

    /// <summary>
    /// <c>model.json</c>, schema <c>jev-model/1</c>. Only the fields the game reads
    /// are declared; JsonUtility skips the rest.
    /// </summary>
    [Serializable]
    public sealed class JevContract
    {
        public const string Schema = "jev-model/1";

        public string schema;
        public string checkpoint;
        public float tick_seconds;
        public GridSpec grid;
        public string[] self_fields;
        public string[] intents;
        public int latent_dim;
        public ProbeSpec[] probes;
        public PositionSpec position_probes;
        public GraphSpecs graphs;

        [Serializable] public sealed class GridSpec { public string[] channels; public int size; public float cell_size; }
        [Serializable] public sealed class ProbeSpec { public string name; public bool display_only; }
        [Serializable] public sealed class PositionSpec { public string x; public string z; public float half_extent; }
        [Serializable] public sealed class GraphSpecs { public ImagineSpec imagine; }
        [Serializable] public sealed class ImagineSpec { public int horizon; }

        public int Horizon => graphs.imagine.horizon;

        public int ProbeIndex(string probe)
        {
            for (int i = 0; i < probes.Length; i++)
                if (probes[i].name == probe) return i;
            throw new InvalidOperationException($"model '{checkpoint}' has no probe '{probe}'");
        }

        /// <summary>
        /// The observation, self-field and intent orders are a model contract
        /// (see CLAUDE.md): a reordered enum would feed the network a silently
        /// scrambled observation. Refuse to run rather than play badly.
        /// </summary>
        public void Validate(string assetName)
        {
            var errors = new List<string>();
            if (schema != Schema) errors.Add($"schema '{schema}', expected '{Schema}'");

            var channels = new string[TacticalObservation.ChannelCount];
            for (int i = 0; i < channels.Length; i++) channels[i] = ((ObservationChannel)i).ToString();
            Compare("grid channels", grid?.channels, channels, errors);
            if (grid != null && grid.size != TacticalObservation.GridSize)
                errors.Add($"grid size {grid.size}, build uses {TacticalObservation.GridSize}");
            if (grid != null && Mathf.Abs(grid.cell_size - TacticalObservation.CellSize) > 1e-4f)
                errors.Add($"cell size {grid.cell_size}, build uses {TacticalObservation.CellSize}");

            Compare("self fields", self_fields, SelfState.FieldNames, errors);

            var names = new string[TacticalIntents.Count];
            for (int i = 0; i < names.Length; i++) names[i] = TacticalIntents.All[i].ToString();
            Compare("intents", intents, names, errors);

            if (Mathf.Abs(tick_seconds - 0.1f) > 1e-4f)
                errors.Add($"trained at {tick_seconds} s per tick, agents decide every 0.1 s");
            if (graphs?.imagine == null || graphs.imagine.horizon <= 0) errors.Add("no imagine horizon");
            if (probes == null || probes.Length == 0) errors.Add("no probes");

            if (errors.Count > 0)
                throw new InvalidOperationException($"{assetName} ({checkpoint}) does not match this build: "
                                                    + string.Join("; ", errors));
        }

        private static void Compare(string what, string[] model, string[] build, List<string> errors)
        {
            if (model == null) { errors.Add($"{what} missing"); return; }
            if (model.Length != build.Length)
            {
                errors.Add($"{what}: model has {model.Length}, build has {build.Length}");
                return;
            }
            for (int i = 0; i < model.Length; i++)
                if (model[i] != build[i])
                    errors.Add($"{what}[{i}]: model '{model[i]}', build '{build[i]}'");
        }
    }
}
