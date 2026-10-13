#pragma once

#include "CoreMinimal.h"
#include "Modules/ModuleManager.h"

PYTORCH_API DECLARE_LOG_CATEGORY_EXTERN(LogTorch, Log, All);

class FPyTorchModule : public IModuleInterface
{
public:
	/** Registers the plugin's Content/Scripts with PythonCLI (unreal_torch, examples) and the environment
	 profile from Project Settings > Plugins > PyTorch. */
	virtual void StartupModule() override;

	/** The plugin's Content/Scripts folder. */
	static FString ScriptsDir();
};
