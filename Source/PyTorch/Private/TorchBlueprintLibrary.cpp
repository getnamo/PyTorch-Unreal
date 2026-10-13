#include "TorchBlueprintLibrary.h"
#include "PyTorch.h"
#include "PythonImageLibrary.h"
#include "Dom/JsonObject.h"
#include "Engine/Texture2D.h"
#include "Engine/TextureRenderTarget2D.h"

namespace TorchLibraryPrivate
{
	static FString OutputName(const FString& Name)
	{
		return Name.IsEmpty() ? TEXT("output") : Name;
	}

	static TSharedPtr<FJsonValue> JsonOutput(const FTorchResult& Result, const FString& Name)
	{
		return Result.Json.IsValid() ? Result.Json->TryGetField(OutputName(Name)) : nullptr;
	}

	static TArray<float> JsonFloats(const TSharedPtr<FJsonValue>& Value)
	{
		TArray<float> Out;
		if (!Value.IsValid())
		{
			return Out;
		}
		double Number = 0.0;
		bool bBool = false;
		const TArray<TSharedPtr<FJsonValue>>* Values = nullptr;
		if (Value->TryGetNumber(Number))
		{
			Out.Add((float)Number);
		}
		else if (Value->TryGetBool(bBool))
		{
			Out.Add(bBool ? 1.f : 0.f);
		}
		else if (Value->TryGetArray(Values))
		{
			for (const TSharedPtr<FJsonValue>& V : *Values)
			{
				Out.Append(JsonFloats(V));
			}
		}
		return Out;
	}
}

bool UTorchBlueprintLibrary::GetOutputTensor(const FTorchResult& Result, const FString& Name, FPyTensor& Tensor)
{
	if (const FPyTensor* Found = Result.Outputs.Find(TorchLibraryPrivate::OutputName(Name)))
	{
		Tensor = *Found;
		return true;
	}
	Tensor = FPyTensor();
	return false;
}

TArray<float> UTorchBlueprintLibrary::GetOutputFloats(const FTorchResult& Result, const FString& Name)
{
	if (const FPyTensor* Found = Result.Outputs.Find(TorchLibraryPrivate::OutputName(Name)))
	{
		return Found->ToFloats();
	}
	return TorchLibraryPrivate::JsonFloats(TorchLibraryPrivate::JsonOutput(Result, Name));
}

float UTorchBlueprintLibrary::GetOutputFloat(const FTorchResult& Result, const FString& Name, int32 Index, float Default)
{
	const TArray<float> Values = GetOutputFloats(Result, Name);
	return Values.IsValidIndex(Index) ? Values[Index] : Default;
}

int32 UTorchBlueprintLibrary::GetOutputInt(const FTorchResult& Result, const FString& Name, int32 Index, int32 Default)
{
	const TArray<float> Values = GetOutputFloats(Result, Name);
	return Values.IsValidIndex(Index) ? FMath::RoundToInt(Values[Index]) : Default;
}

int32 UTorchBlueprintLibrary::GetOutputArgmax(const FTorchResult& Result, const FString& Name)
{
	const TArray<float> Values = GetOutputFloats(Result, Name);
	int32 Best = -1;
	for (int32 i = 0; i < Values.Num(); ++i)
	{
		if (Best < 0 || Values[i] > Values[Best])
		{
			Best = i;
		}
	}
	return Best;
}

bool UTorchBlueprintLibrary::GetOutputString(const FTorchResult& Result, const FString& Name, FString& Value)
{
	const TSharedPtr<FJsonValue> V = TorchLibraryPrivate::JsonOutput(Result, Name);
	if (!V.IsValid())
	{
		return false;
	}
	if (V->Type == EJson::String || V->Type == EJson::Number || V->Type == EJson::Boolean)
	{
		Value = V->AsString();
		return true;
	}
	Value = PythonWire::ToJsonString(V);
	return true;
}

TArray<FString> UTorchBlueprintLibrary::GetOutputNames(const FTorchResult& Result)
{
	TArray<FString> Names;
	Result.Outputs.GetKeys(Names);
	if (Result.Json.IsValid())
	{
		for (const TPair<FString, TSharedPtr<FJsonValue>>& Field : Result.Json->Values)
		{
			Names.AddUnique(Field.Key);
		}
	}
	return Names;
}

bool UTorchBlueprintLibrary::WriteOutputToRenderTarget(const FTorchResult& Result, const FString& Name, UTextureRenderTarget2D* RenderTarget, bool bChannelsFirst)
{
	FPyTensor Tensor;
	if (!GetOutputTensor(Result, Name, Tensor))
	{
		UE_LOG(LogTorch, Warning, TEXT("WriteOutputToRenderTarget: no tensor output '%s'"), *TorchLibraryPrivate::OutputName(Name));
		return false;
	}
	return UPythonImageLibrary::WriteTensorToRenderTarget(Tensor, RenderTarget, bChannelsFirst);
}

UTexture2D* UTorchBlueprintLibrary::OutputToTexture(const FTorchResult& Result, const FString& Name, bool bChannelsFirst)
{
	FPyTensor Tensor;
	return GetOutputTensor(Result, Name, Tensor) ? UPythonImageLibrary::TensorToTexture(Tensor, bChannelsFirst) : nullptr;
}

FPyTensor UTorchBlueprintLibrary::MakeImageTensorFromTexture(UTexture2D* Texture)
{
	FPyImageOptions Options;
	Options.Channels = EPyImageChannels::RGBA;
	Options.Normalization = EPyImageNormalization::Raw;
	FPyTensor F = UPythonImageLibrary::TextureToTensor(Texture, Options);
	//Raw 0..255 floats -> uint8: a quarter of the data, converted on the model's device
	FPyTensor Out;
	Out.DType = EPyDType::UInt8;
	Out.Shape = F.Shape;
	const TArray<float> Values = F.ToFloats();
	Out.Data.SetNumUninitialized(Values.Num());
	for (int32 i = 0; i < Values.Num(); ++i)
	{
		Out.Data[i] = (uint8)FMath::Clamp(FMath::RoundToInt(Values[i]), 0, 255);
	}
	return Out;
}

FPyTensor UTorchBlueprintLibrary::MakeImageTensorFromRenderTarget(UTextureRenderTarget2D* RenderTarget)
{
	FPyImageOptions Options;
	Options.Channels = EPyImageChannels::RGBA;
	Options.Normalization = EPyImageNormalization::Raw;
	const TArray<float> Values = UPythonImageLibrary::RenderTargetToTensor(RenderTarget, Options).ToFloats();
	FPyTensor Out;
	Out.DType = EPyDType::UInt8;
	if (RenderTarget && Values.Num() == RenderTarget->SizeX * RenderTarget->SizeY * 4)
	{
		Out.Shape = { RenderTarget->SizeY, RenderTarget->SizeX, 4 };
		Out.Data.SetNumUninitialized(Values.Num());
		for (int32 i = 0; i < Values.Num(); ++i)
		{
			Out.Data[i] = (uint8)FMath::Clamp(FMath::RoundToInt(Values[i]), 0, 255);
		}
	}
	return Out;
}

FPyTensor UTorchBlueprintLibrary::MakeTensorFromVector(FVector Value)
{
	return FPyTensor::FromFloats({ (float)Value.X, (float)Value.Y, (float)Value.Z }, { 3 });
}
