#include "TorchModel.h"
#include "PyTorch.h"
#include "TorchSubsystem.h"
#include "Dom/JsonObject.h"

FString UTorchModel::GetModelName() const
{
	return ModelName.IsEmpty() ? GetName() : ModelName;
}

const FTorchMethodSpec* UTorchModel::FindMethod(const FString& Name) const
{
	return Methods.FindByPredicate([&Name](const FTorchMethodSpec& M) { return M.Name == Name; });
}

TSharedRef<FJsonObject> UTorchModel::MakeLoadOptions() const
{
	TSharedRef<FJsonObject> O = MakeShared<FJsonObject>();
	switch (Device)
	{
	case ETorchDevice::Auto: O->SetStringField(TEXT("device"), TEXT("auto")); break;
	case ETorchDevice::CUDA: O->SetStringField(TEXT("device"), TEXT("cuda")); break;
	case ETorchDevice::CPU: O->SetStringField(TEXT("device"), TEXT("cpu")); break;
	default: break;
	}
	switch (Precision)
	{
	case ETorchPrecision::FP32: O->SetStringField(TEXT("precision"), TEXT("fp32")); break;
	case ETorchPrecision::BF16: O->SetStringField(TEXT("precision"), TEXT("bf16")); break;
	case ETorchPrecision::FP16: O->SetStringField(TEXT("precision"), TEXT("fp16")); break;
	default: break;
	}
	switch (Compile)
	{
	case ETorchCompile::Off: O->SetBoolField(TEXT("compile"), false); break;
	case ETorchCompile::On: O->SetStringField(TEXT("compile"), TEXT("default")); break;
	case ETorchCompile::ReduceOverhead: O->SetStringField(TEXT("compile"), TEXT("reduce-overhead")); break;
	default: break;
	}
	O->SetBoolField(TEXT("warmup"), bWarmup);
	return O;
}

void UTorchModel::ApplySignature(const TSharedPtr<FJsonObject>& Signature, const TArray<TSharedPtr<FJsonValue>>* JobList)
{
	if (Signature.IsValid())
	{
		Methods.Reset();
		const TArray<TSharedPtr<FJsonValue>>* MethodValues = nullptr;
		if (Signature->TryGetArrayField(TEXT("methods"), MethodValues))
		{
			for (const TSharedPtr<FJsonValue>& MV : *MethodValues)
			{
				const TSharedPtr<FJsonObject> M = MV->AsObject();
				FTorchMethodSpec& Spec = Methods.AddDefaulted_GetRef();
				Spec.Name = M->GetStringField(TEXT("name"));
				M->TryGetBoolField(TEXT("batch"), Spec.bBatched);
				M->TryGetStringField(TEXT("doc"), Spec.Doc);
				const TArray<TSharedPtr<FJsonValue>>* Inputs = nullptr;
				if (M->TryGetArrayField(TEXT("inputs"), Inputs))
				{
					for (const TSharedPtr<FJsonValue>& IV : *Inputs)
					{
						TSharedPtr<FJsonObject> I = IV->AsObject();
						FTorchInputSpec& In = Spec.Inputs.AddDefaulted_GetRef();
						In.Name = I->GetStringField(TEXT("name"));
						In.Kind = I->GetStringField(TEXT("kind"));
						I->TryGetBoolField(TEXT("required"), In.bRequired);
						if (!I->TryGetStringField(TEXT("dtype"), In.Type))
						{
							I->TryGetStringField(TEXT("type"), In.Type);
						}
						const TArray<TSharedPtr<FJsonValue>>* Shape = nullptr;
						if (I->TryGetArrayField(TEXT("shape"), Shape) || I->TryGetArrayField(TEXT("size"), Shape))
						{
							for (const TSharedPtr<FJsonValue>& D : *Shape)
							{
								In.Shape.Add((int64)D->AsNumber());
							}
						}
						TSharedRef<FJsonObject> Details = MakeShared<FJsonObject>();
						for (const TPair<FString, TSharedPtr<FJsonValue>>& Field : I->Values)
						{
							if (Field.Key != TEXT("name") && Field.Key != TEXT("kind") && Field.Key != TEXT("required") && Field.Key != TEXT("dtype")
								&& Field.Key != TEXT("type") && Field.Key != TEXT("shape") && Field.Key != TEXT("size"))
							{
								Details->SetField(Field.Key, Field.Value);
							}
						}
						In.Details = Details->Values.Num() ? PythonWire::ToJsonString(MakeShared<FJsonValueObject>(Details)) : FString();
					}
				}
			}
		}
	}
	if (JobList)
	{
		Jobs.Reset();
		for (const TSharedPtr<FJsonValue>& JV : *JobList)
		{
			const TSharedPtr<FJsonObject> J = JV->AsObject();
			FTorchJobSpec& Spec = Jobs.AddDefaulted_GetRef();
			Spec.Name = J->GetStringField(TEXT("name"));
			J->TryGetStringField(TEXT("doc"), Spec.Doc);
			Spec.ParamsJson = PythonWire::ToJsonString(J->TryGetField(TEXT("params")));
		}
	}
}

#if WITH_EDITOR
void UTorchModel::RefreshSignature()
{
	UTorchSubsystem* Torch = UTorchSubsystem::Get();
	if (!Torch)
	{
		return;
	}
	TWeakObjectPtr<UTorchModel> Weak(this);
	Torch->FetchSignature(this, [Weak](bool bOk, const FString& Error)
	{
		if (UTorchModel* Self = Weak.Get())
		{
			if (bOk)
			{
				Self->Modify();
				UE_LOG(LogTorch, Display, TEXT("%s: %d methods, %d jobs"), *Self->GetName(), Self->Methods.Num(), Self->Jobs.Num());
			}
			else
			{
				UE_LOG(LogTorch, Warning, TEXT("%s: refreshing the signature failed: %s"), *Self->GetName(), *Error);
			}
		}
	});
}
#endif
