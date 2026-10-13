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

		// unreal_torch, the examples and the environment profiles go with packaged builds.
		RuntimeDependencies.Add(Path.Combine(PluginDirectory, "Content", "Scripts", "..."));
	}
}
