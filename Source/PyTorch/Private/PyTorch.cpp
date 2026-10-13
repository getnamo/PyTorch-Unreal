#include "PyTorch.h"
#include "PythonEnvironment.h"
#include "Interfaces/IPluginManager.h"
#include "Misc/Paths.h"

DEFINE_LOG_CATEGORY(LogTorch);

FString FPyTorchModule::ScriptsDir()
{
	const TSharedPtr<IPlugin> Plugin = IPluginManager::Get().FindPlugin(TEXT("PyTorch"));
	return Plugin.IsValid() ? FPaths::ConvertRelativePathToFull(Plugin->GetBaseDir() / TEXT("Content/Scripts")) : FString();
}

void FPyTorchModule::StartupModule()
{
	//Before the Python environment is set up (the engine subsystem initializes after Default-phase modules)
	FPythonEnvironment::RegisterScriptDirectory(ScriptsDir());
}

IMPLEMENT_MODULE(FPyTorchModule, PyTorch)
