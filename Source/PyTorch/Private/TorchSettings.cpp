#include "TorchSettings.h"
#include "PyTorch.h"
#include "PythonCLISettings.h"
#include "PythonEnvironment.h"
#include "PythonSubsystem.h"
#include "Misc/ConfigCacheIni.h"
#include "Misc/Paths.h"

namespace TorchSettingsPrivate
{
	static FString Requirements(const TCHAR* Name)
	{
		return FPyTorchModule::ScriptsDir() / TEXT("requirements") / Name;
	}

	static const TCHAR* AllFiles[] = { TEXT("torch-cuda.txt"), TEXT("torch-cpu.txt"), TEXT("torchvision.txt"), TEXT("torchaudio.txt"),
		TEXT("onnx.txt"), TEXT("gymnasium.txt"), TEXT("tensorboard.txt"), TEXT("huggingface.txt"), TEXT("packaged-cpu.txt") };

	/** The project's own Torch Backend setting (config), as opposed to a value another plugin set in memory. */
	static FString ConfiguredTorchBackend()
	{
		FString Value;
		if (GConfig)
		{
			GConfig->GetString(TEXT("/Script/PythonCLI.PythonCLISettings"), TEXT("TorchBackend"), Value, GGameIni);
		}
		return Value;
	}
}

UTorchSettings::UTorchSettings()
{
	SectionName = TEXT("PyTorch");
}

FString UTorchSettings::TorchBackend(ETorchProfile InProfile)
{
	switch (InProfile)
	{
	case ETorchProfile::CUDAAuto: return TEXT("auto");
	case ETorchProfile::CUDA130: return TEXT("cu130");
	case ETorchProfile::CUDA132: return TEXT("cu132");
	case ETorchProfile::CPU: return TEXT("cpu");
	case ETorchProfile::ROCm: return TEXT("auto"); //uv detects AMD GPUs and picks the ROCm build
	case ETorchProfile::XPU: return TEXT("xpu");
	default: return FString();
	}
}

void UTorchSettings::ApplyProfile() const
{
	using namespace TorchSettingsPrivate;
	for (const TCHAR* File : AllFiles)
	{
		FPythonEnvironment::UnregisterRequirementsFile(Requirements(File));
		FPythonEnvironment::UnregisterPackagedRequirementsFile(Requirements(File));
	}
	FPythonEnvironment::SetDefaultPackagedTorchBackend(FString());
	if (Profile == ETorchProfile::None)
	{
		return;
	}

	const bool bCuda = Profile == ETorchProfile::CUDAAuto || Profile == ETorchProfile::CUDA130 || Profile == ETorchProfile::CUDA132;
	const FString TorchFile = Requirements(bCuda ? TEXT("torch-cuda.txt") : TEXT("torch-cpu.txt"));
	FPythonEnvironment::RegisterRequirementsFile(TorchFile);
	if (bTorchvision) { FPythonEnvironment::RegisterRequirementsFile(Requirements(TEXT("torchvision.txt"))); }
	if (bTorchaudio) { FPythonEnvironment::RegisterRequirementsFile(Requirements(TEXT("torchaudio.txt"))); }
	if (bOnnxExport) { FPythonEnvironment::RegisterRequirementsFile(Requirements(TEXT("onnx.txt"))); }
	if (bGymnasium) { FPythonEnvironment::RegisterRequirementsFile(Requirements(TEXT("gymnasium.txt"))); }
	if (bTensorBoard) { FPythonEnvironment::RegisterRequirementsFile(Requirements(TEXT("tensorboard.txt"))); }
	if (bHuggingFaceHub) { FPythonEnvironment::RegisterRequirementsFile(Requirements(TEXT("huggingface.txt"))); }

	//The project's own Torch Backend setting wins; otherwise the profile's (in memory only)
	if (ConfiguredTorchBackend().IsEmpty())
	{
		GetMutableDefault<UPythonCLISettings>()->TorchBackend = TorchBackend(Profile);
	}

	if (PackagedProfile == ETorchPackagedProfile::CPU)
	{
		//Packaged builds: CPU torch instead of the editor's torch file (the extras stay, as CPU builds)
		FPythonEnvironment::RegisterPackagedRequirementsFile(Requirements(TEXT("packaged-cpu.txt")), TorchFile);
		FPythonEnvironment::SetDefaultPackagedTorchBackend(TEXT("cpu"));
	}

	//torch.compile's caches in the project, so compiled kernels survive editor restarts
	const FString Cache = FPythonEnvironment::IsPackagedBuild()
		? FPaths::ConvertRelativePathToFull(FPaths::ProjectSavedDir() / TEXT("Torch/CompileCache"))
		: FPaths::ConvertRelativePathToFull(FPaths::ProjectDir() / CompileCacheDirectory);
	FPythonEnvironment::RegisterEnvironmentVariable(TEXT("TORCHINDUCTOR_CACHE_DIR"), Cache / TEXT("inductor"));
	FPythonEnvironment::RegisterEnvironmentVariable(TEXT("TRITON_CACHE_DIR"), Cache / TEXT("triton"));

	//Pretrained weights (torchvision, torch.hub) under the project unless the machine has its own TORCH_HOME
	if (FPlatformMisc::GetEnvironmentVariable(TEXT("TORCH_HOME")).IsEmpty())
	{
		FPythonEnvironment::RegisterEnvironmentVariable(TEXT("TORCH_HOME"), FPaths::ConvertRelativePathToFull(FPaths::ProjectSavedDir() / TEXT("Torch/Hub")));
	}
}

#if WITH_EDITOR
void UTorchSettings::PostEditChangeProperty(FPropertyChangedEvent& PropertyChangedEvent)
{
	Super::PostEditChangeProperty(PropertyChangedEvent);
	ApplyProfile();
	if (UPythonSubsystem* Subsystem = UPythonSubsystem::Get())
	{
		Subsystem->SetupEnvironment(); //Installs the new profile in the background; running processes keep theirs
	}
}
#endif
