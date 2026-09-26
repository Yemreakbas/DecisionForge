using System.IO;
using UnityEditor;
using UnityEditor.Build.Reporting;
using UnityEngine;

namespace JevNpcBrain.EditorTools
{
    /// <summary>
    /// Builds the headless data collector: the arena scene as a Windows player that
    /// Training/collect.ps1 launches with -batchmode -nographics -collect.
    ///
    /// A player rather than the editor in batch mode, because the editor locks the
    /// project folder: a player runs while you keep working in Unity, and several
    /// of them run side by side, one seed each.
    /// </summary>
    public static class CollectorBuild
    {
        public const string ScenePath = "Assets/_Project/Scenes/01_Octagon.unity";
        public const string OutputPath = "Builds/Collector/JevCollector.exe";

        [MenuItem("JEV/Build Data Collector")]
        public static void Build()
        {
            // A dirty open scene makes Unity stop the build on a modal "Scene(s) Have
            // Been Modified" dialog -- invisible to anyone driving the editor over MCP,
            // which then waits on a build that never starts. Refuse instead, loudly.
            for (int i = 0; i < UnityEditor.SceneManagement.EditorSceneManager.sceneCount; i++)
            {
                var scene = UnityEditor.SceneManagement.EditorSceneManager.GetSceneAt(i);
                if (!scene.isDirty) continue;
                Debug.LogError($"[CollectorBuild] Build refused: '{scene.path}' has unsaved changes. " +
                               "Save or revert it first -- the player is built from the scene on disk.");
                return;
            }

            var report = BuildPipeline.BuildPlayer(new BuildPlayerOptions
            {
                scenes = new[] { ScenePath },
                locationPathName = OutputPath,
                target = BuildTarget.StandaloneWindows64,
                options = BuildOptions.None
            });

            var summary = report.summary;
            if (summary.result == BuildResult.Succeeded)
                Debug.Log($"[CollectorBuild] {Path.GetFullPath(OutputPath)} " +
                          $"({summary.totalSize / (1024 * 1024)} MB, {summary.totalTime.TotalSeconds:F0} s)");
            else
                Debug.LogError($"[CollectorBuild] Build {summary.result} with {summary.totalErrors} error(s).");
        }
    }
}
