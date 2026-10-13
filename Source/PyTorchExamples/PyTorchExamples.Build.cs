using UnrealBuildTool;

// Classes for the plugin's example maps (reinforcement learning tasks). Not needed by projects using PyTorch.
public class PyTorchExamples : ModuleRules
{
	public PyTorchExamples(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

		PublicDependencyModuleNames.AddRange(new string[]
		{
			"Core",
			"CoreUObject",
			"Engine",
			"PyTorch",
			"PythonCLI",
		});

		PrivateDependencyModuleNames.AddRange(new string[] { "Json", "InputCore" });
		// The PyTorch module's test helpers (FEnv, WaitUntil, ...)
		PrivateIncludePaths.Add(System.IO.Path.Combine(ModuleDirectory, "..", "PyTorch", "Private", "Tests"));

		if (Target.bBuildEditor)
		{
			PrivateDependencyModuleNames.AddRange(new string[] { "UnrealEd", "RHI" }); // the example-map play test (PIE)
		}
	}
}
