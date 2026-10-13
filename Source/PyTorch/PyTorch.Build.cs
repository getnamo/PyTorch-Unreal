using System.IO;
using UnrealBuildTool;

public class PyTorch : ModuleRules
{
	public PyTorch(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

		PublicDependencyModuleNames.AddRange(new string[]
		{
			"Core",
			"CoreUObject",
			"Engine",
			"DeveloperSettings",
			"PythonCLI",
			"NNE",
		});

		PrivateDependencyModuleNames.AddRange(new string[]
		{
			"Projects",
			"Json",
			"AssetRegistry",
			"RenderCore",
			"RHI",
		});

		if (Target.bBuildEditor)
		{
			PrivateDependencyModuleNames.AddRange(new string[] { "UnrealEd", "InputCore" }); // the example-map play test (PIE)
		}

		// unreal_torch, the examples and the environment profiles go with packaged builds.
		RuntimeDependencies.Add(Path.Combine(PluginDirectory, "Content", "Scripts", "..."));

		// Weights (load_weights looks in Content/Models/): the examples' and the project's.
		RuntimeDependencies.Add(Path.Combine(PluginDirectory, "Content", "Models", "..."));
		if (Target.ProjectFile != null)
		{
			string ProjectModels = Path.Combine(Target.ProjectFile.Directory.FullName, "Content", "Models");
			if (Directory.Exists(ProjectModels))
			{
				RuntimeDependencies.Add(Path.Combine(ProjectModels, "..."));
			}
		}
	}
}
