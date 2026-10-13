#pragma once

#include "CoreMinimal.h"
#include "Kismet/BlueprintFunctionLibrary.h"
#include "TorchTypes.h"
#include "TorchBlueprintLibrary.generated.h"

class UTextureRenderTarget2D;
class UTexture2D;

/** Reading Run Model results, and making inputs. */
UCLASS()
class PYTORCH_API UTorchBlueprintLibrary : public UBlueprintFunctionLibrary
{
	GENERATED_BODY()
public:
	/** A tensor output ("output" when the method returns one tensor). */
	UFUNCTION(BlueprintPure, Category = "Torch|Result", meta = (AutoCreateRefTerm = "Name"))
	static bool GetOutputTensor(const FTorchResult& Result, const FString& Name, FPyTensor& Tensor);

	/** An output as floats: a tensor's values, or a JSON number / array of numbers. */
	UFUNCTION(BlueprintPure, Category = "Torch|Result", meta = (AutoCreateRefTerm = "Name"))
	static TArray<float> GetOutputFloats(const FTorchResult& Result, const FString& Name);

	/** One value of an output (a tensor's element at Index, or a JSON number), or Default. */
	UFUNCTION(BlueprintPure, Category = "Torch|Result", meta = (AutoCreateRefTerm = "Name"))
	static float GetOutputFloat(const FTorchResult& Result, const FString& Name, int32 Index = 0, float Default = 0.f);

	UFUNCTION(BlueprintPure, Category = "Torch|Result", meta = (AutoCreateRefTerm = "Name"))
	static int32 GetOutputInt(const FTorchResult& Result, const FString& Name, int32 Index = 0, int32 Default = 0);

	/** The index of the largest value of an output (e.g. the class of logits / probabilities), or -1. */
	UFUNCTION(BlueprintPure, Category = "Torch|Result", meta = (AutoCreateRefTerm = "Name"))
	static int32 GetOutputArgmax(const FTorchResult& Result, const FString& Name);

	/** A string output (JSON), or a number / bool as text. */
	UFUNCTION(BlueprintPure, Category = "Torch|Result", meta = (AutoCreateRefTerm = "Name"))
	static bool GetOutputString(const FTorchResult& Result, const FString& Name, FString& Value);

	/** The names of the result's outputs (tensors and JSON values). */
	UFUNCTION(BlueprintPure, Category = "Torch|Result")
	static TArray<FString> GetOutputNames(const FTorchResult& Result);

	/** An image output -> a render target (resized to the image), e.g. a stylized frame. (H, W, C) or (C, H, W). */
	UFUNCTION(BlueprintCallable, Category = "Torch|Result", meta = (AutoCreateRefTerm = "Name"))
	static bool WriteOutputToRenderTarget(const FTorchResult& Result, const FString& Name, UTextureRenderTarget2D* RenderTarget, bool bChannelsFirst = true);

	/** An image output -> a new transient texture. */
	UFUNCTION(BlueprintCallable, Category = "Torch|Result", meta = (AutoCreateRefTerm = "Name"))
	static UTexture2D* OutputToTexture(const FTorchResult& Result, const FString& Name, bool bChannelsFirst = true);

	/** Pixels of a texture as an (H, W, 4) uint8 RGBA tensor: the model's ut.Image annotation converts (resize,
	 gray, normalization) on its device. */
	UFUNCTION(BlueprintPure, Category = "Torch|Input")
	static FPyTensor MakeImageTensorFromTexture(UTexture2D* Texture);

	/** As MakeImageTensorFromTexture, reading the render target now (blocking; use a stream for every frame). */
	UFUNCTION(BlueprintCallable, Category = "Torch|Input")
	static FPyTensor MakeImageTensorFromRenderTarget(UTextureRenderTarget2D* RenderTarget);

	/** A float tensor from a vector (3 values). */
	UFUNCTION(BlueprintPure, Category = "Torch|Input")
	static FPyTensor MakeTensorFromVector(FVector Value);
};
