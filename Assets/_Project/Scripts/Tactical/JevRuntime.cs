using System;
using Unity.InferenceEngine;

namespace JevNpcBrain.Tactical
{
    using Core;
    using Perception;

    /// <summary>
    /// The world model loaded into Inference Engine workers, shared by every
    /// <see cref="JevBrain"/> in a match. Agents decide one after another on the
    /// main thread, so one encoder worker and one imagine worker serve them all.
    ///
    /// One call to <see cref="Imagine"/> is the planner's whole inference budget
    /// for a tick: encode the observation, then roll every candidate intent
    /// sequence forward <see cref="Horizon"/> ticks in latent space and read the
    /// probes at each imagined step.
    ///
    /// The candidate set is fixed -- candidate c holds intent c for the whole
    /// horizon, and the brain replans every tick -- so its one-hot tensor is
    /// uploaded once and never touched again.
    /// </summary>
    public sealed class JevRuntime : IDisposable
    {
        public readonly JevContract Contract;
        public readonly int Horizon;
        public readonly int Candidates = TacticalIntents.Count;
        public readonly int ProbeCount;

        /// <summary>Probe outputs of the last <see cref="Imagine"/>, laid out (candidate, step, probe).</summary>
        public readonly float[] Probes;

        private readonly Worker _encoder;
        private readonly Worker _imagine;
        private readonly Tensor<float> _grid;
        private readonly Tensor<float> _self;
        private readonly Tensor<float> _latent;
        private readonly Tensor<float> _intents;
        private readonly float[] _latentBuffer;
        private readonly bool _cpu;

        public JevRuntime(JevModelAsset asset, BackendType backend = BackendType.CPU)
        {
            _cpu = backend == BackendType.CPU;
            Contract = asset.LoadContract();
            Horizon = Contract.Horizon;
            ProbeCount = Contract.probes.Length;
            Probes = new float[Candidates * Horizon * ProbeCount];
            _latentBuffer = new float[Contract.latent_dim];

            _encoder = new Worker(ModelLoader.Load(asset.Encoder), backend);
            _imagine = new Worker(ModelLoader.Load(asset.Imagine), backend);

            const int s = TacticalObservation.GridSize;
            _grid = new Tensor<float>(new TensorShape(1, TacticalObservation.ChannelCount, s, s),
                new float[TacticalObservation.GridLength]);
            _self = new Tensor<float>(new TensorShape(1, SelfState.Length), new float[SelfState.Length]);
            _latent = new Tensor<float>(new TensorShape(1, Contract.latent_dim), _latentBuffer);

            var onehot = new float[Candidates * Horizon * Candidates];
            for (int c = 0; c < Candidates; c++)
                for (int k = 0; k < Horizon; k++)
                    onehot[(c * Horizon + k) * Candidates + c] = 1f;
            _intents = new Tensor<float>(new TensorShape(Candidates, Horizon, Candidates), onehot);
        }

        /// <summary>Encodes <paramref name="obs"/> and imagines every candidate into <see cref="Probes"/>.</summary>
        public void Imagine(TacticalObservation obs)
        {
            _grid.Upload(obs.Grid);
            _self.Upload(obs.SelfVector);
            _encoder.SetInput("grid", _grid);
            _encoder.SetInput("self", _self);
            _encoder.Schedule();

            Read((Tensor<float>)_encoder.PeekOutput("latent"), _latentBuffer);
            _latent.Upload(_latentBuffer);

            _imagine.SetInput("latent", _latent);
            _imagine.SetInput("intents", _intents);
            _imagine.Schedule();

            Read((Tensor<float>)_imagine.PeekOutput("probes"), Probes);
        }

        /// <summary>
        /// CPU tensors are read in place, without allocating; GPU tensors have to
        /// be downloaded first, which allocates -- acceptable on that path only.
        /// </summary>
        private void Read(Tensor<float> tensor, float[] destination)
        {
            if (_cpu)
            {
                tensor.CompleteAllPendingOperations();
                tensor.AsReadOnlySpan().CopyTo(destination);
            }
            else
            {
                var data = tensor.DownloadToArray();
                Array.Copy(data, destination, data.Length);
            }
        }

        public void Dispose()
        {
            _encoder?.Dispose();
            _imagine?.Dispose();
            _grid?.Dispose();
            _self?.Dispose();
            _latent?.Dispose();
            _intents?.Dispose();
        }
    }
}
