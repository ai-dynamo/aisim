// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// SPDX-License-Identifier: Apache-2.0

//! Stock class names from vLLM 0.31–0.33. A registered class can stop native
//! dispatch even when AISim does not project that class’s physical attention.
//! See THIRD_PARTY_NOTICES.md for the pinned model registries.

use super::heads::Decoder;

#[derive(Clone, Copy, PartialEq)]
pub(super) enum Registration {
    Decoder(Decoder),
    Qwen2Vl,
    Other,
}

impl Registration {
    pub(super) fn decoder(self) -> Option<Decoder> {
        match self {
            Self::Decoder(decoder) => Some(decoder),
            Self::Qwen2Vl => Some(Decoder::Qwen2),
            Self::Other => None,
        }
    }
}

/// A versionless archive establishes a result only if every supported registry agrees.
pub(super) fn at_release(
    version: Option<&str>,
    lookup: impl Fn(u8) -> Option<Registration>,
) -> Option<Registration> {
    match version.and_then(|s| s.split('+').next()) {
        Some("0.31.0") => lookup(31),
        Some("0.32.0") => lookup(32),
        Some("0.33.0") => lookup(33),
        _ => {
            let first = lookup(31)?;
            [32, 33]
                .into_iter()
                .all(|minor| lookup(minor) == Some(first))
                .then_some(first)
        }
    }
}

/// With no literal stock class the auto loader can choose Transformers before
/// native normalization. Once a stock name is present, even a normalized first
/// alternative takes precedence over a later exact entry.
pub(super) fn select<'a>(
    names: impl Iterator<Item = Option<&'a str>> + Clone,
    release: Option<&str>,
    require_native: bool,
    stock_model_lookup: bool,
) -> Option<Registration> {
    at_release(release, |minor| {
        if !require_native && !names.clone().flatten().any(|n| lookup(n, minor).is_some()) {
            // Neither pinned Transformers release exports the nonliteral generation
            // aliases that normalize to our six native decoders. Without a dynamic
            // AutoModel mapping its lookup fails and vLLM continues with these names.
            let first = names.clone().next().flatten()?;
            if !stock_model_lookup || normalize(first, minor)?.decoder().is_none() {
                return None;
            }
        }
        for name in names.clone() {
            if let Some(class) = normalize(name?, minor) {
                return Some(class);
            }
        }
        None
    })
}

/// The native registry tries an exact class first. For generation it can
/// substitute a registered base class for one of the four generation suffixes.
pub(super) fn normalize(name: &str, minor: u8) -> Option<Registration> {
    const SUFFIXES: &[&str] = &[
        "ForCausalLM",
        "ForConditionalGeneration",
        "ChatModel",
        "LMHeadModel",
        "ForTextEncoding",
        "EmbeddingModel",
        "ForSequenceClassification",
        "ForTokenClassification",
        "ForAudioClassification",
        "ForImageClassification",
        "ForVideoClassification",
        "ClassificationModel",
        "ForRewardModeling",
        "RewardModel",
        "Model",
    ];
    lookup(name, minor).or_else(|| {
        let suffix = SUFFIXES[..4]
            .iter()
            .find(|suffix| name.ends_with(**suffix))?;
        SUFFIXES
            .iter()
            .find_map(|other| lookup(&name.replace(suffix, other), minor))
    })
}

// Other stock registrations shared by all three releases, in lexical order.
#[rustfmt::skip]
const OTHER: &[&str] = &[
    "AXK1ForCausalLM", "AfmoeForCausalLM", "ApertusForCausalLM", "ArceeForCausalLM",
    "AriaForConditionalGeneration", "AudioFlamingo3ForConditionalGeneration",
    "BagelForConditionalGeneration", "BailingMoeForCausalLM", "BailingMoeV25MTPModel",
    "BailingMoeV2ForCausalLM", "BailingMoeV2_5ForCausalLM", "BailingMoeV3ForCausalLM",
    "BailingMoeV3MTPModel", "BailingMoeV3VLForConditionalGeneration", "BeeForConditionalGeneration",
    "BertForMaskedLM", "BertForSequenceClassification", "BertForTokenClassification", "BertModel",
    "BertSpladeSparseEmbeddingModel", "BgeM3EmbeddingModel", "Blip2ForConditionalGeneration",
    "BloomForCausalLM", "CLIPModel", "ChatGLMForConditionalGeneration", "ChatGLMModel",
    "Cohere2ForCausalLM", "Cohere2MoeForCausalLM", "Cohere2VisionForConditionalGeneration",
    "CohereAsrForConditionalGeneration", "CohereCompassForConditionalGeneration",
    "CohereForCausalLM", "ColBERTJinaRobertaModel", "ColBERTLfm2Model", "ColBERTModernBertModel",
    "ColModernVBertForRetrieval", "ColPaliForRetrieval", "ColQwen3", "ColQwen3_5",
    "Cosmos3EdgeForConditionalGeneration", "Cosmos3ForConditionalGeneration", "DFlash2DraftModel",
    "DFlashDraftModel", "DFlashLagunaForCausalLM", "DFlashMuseGlimmerAssistantModel",
    "DSparkDraftModel", "DSparkV41DraftModel", "DbrxForCausalLM", "DeciLMForCausalLM",
    "DeepSeekMTPModel", "DeepSeekV4MTPModel", "DeepseekForCausalLM", "DeepseekOCR2ForCausalLM",
    "DeepseekOCRForCausalLM", "DeepseekV2ForCausalLM", "DeepseekV32ForCausalLM",
    "DeepseekV32MTPModel", "DeepseekV3BidirectionalModel", "DeepseekV3ForCausalLM",
    "DeepseekV41ForCausalLM", "DeepseekV4ForCausalLM", "DeepseekV4ForConditionalGeneration",
    "DeepseekVLV2ForCausalLM", "DiffusionGemmaForBlockDiffusion", "Dots3NoteForCausalLM",
    "Dots3NoteMTPModel", "DotsOCRForCausalLM", "Eagle2_5_VLForConditionalGeneration",
    "Eagle3DeepseekV2ForCausalLM", "Eagle3DeepseekV3ForCausalLM", "Eagle3LlamaForCausalLM",
    "Eagle3MiniMaxM2ForCausalLM", "Eagle3Qwen2_5vlForCausalLM", "Eagle3Qwen3ForCausalLM",
    "Eagle3Qwen3vlForCausalLM", "EagleCohereForCausalLM", "EagleDeepSeekMTPModel",
    "EagleLlama4ForCausalLM", "EagleLlamaForCausalLM", "EagleMiniCPMForCausalLM",
    "EagleMistralForCausalLM", "EagleMistralLarge3ForCausalLM", "Emu3ForConditionalGeneration",
    "Ernie4_5ForCausalLM", "Ernie4_5_MoeForCausalLM", "Ernie4_5_VLMoeForConditionalGeneration",
    "ErnieMTPModel", "Exaone4ForCausalLM", "Exaone4_5_ForConditionalGeneration", "Exaone4_5_MTP",
    "ExaoneForCausalLM", "ExaoneMoeForCausalLM", "ExaoneMoeMTP", "ExtractHiddenStatesModel",
    "FalconH1ForCausalLM", "FalconMambaForCausalLM", "FireRedASR2ForConditionalGeneration",
    "FlexOlmoForCausalLM", "FunASRForConditionalGeneration", "FunAudioChatForConditionalGeneration",
    "GLM4VForCausalLM", "GPT2ForSequenceClassification", "GPT2LMHeadModel", "GPTBigCodeForCausalLM",
    "GPTJForCausalLM", "GPTNeoXForCausalLM", "Gemma2ForCausalLM", "Gemma2Model",
    "Gemma3ForCausalLM", "Gemma3ForConditionalGeneration", "Gemma3TextModel", "Gemma3nForCausalLM",
    "Gemma3nForConditionalGeneration", "Gemma4DSparkModel", "Gemma4ForCausalLM",
    "Gemma4ForConditionalGeneration", "Gemma4MTPModel", "Gemma4UnifiedForConditionalGeneration",
    "GemmaForCausalLM", "Glm4ForCausalLM", "Glm4MoeForCausalLM", "Glm4MoeLiteForCausalLM",
    "Glm4MoeLiteMTPModel", "Glm4MoeMTPModel", "Glm4vForConditionalGeneration",
    "Glm4vMoeForConditionalGeneration", "Glm5NextForCausalLM", "Glm5NextForConditionalGeneration",
    "Glm5NextMTPModel", "GlmAsrForConditionalGeneration", "GlmForCausalLM", "GlmMoeDsaForCausalLM",
    "GlmOcrForConditionalGeneration", "GlmOcrMTPModel", "GptOssForCausalLM",
    "Granite4VisionForConditionalGeneration", "GraniteForCausalLM", "GraniteMoeForCausalLM",
    "GraniteMoeHybridForCausalLM", "GraniteMoeSWAForCausalLM", "GraniteMoeSharedForCausalLM",
    "GraniteSWAForCausalLM", "GraniteSpeechForConditionalGeneration",
    "GraniteSpeechPlusForConditionalGeneration", "GteModel", "GteNewForSequenceClassification",
    "GteNewModel", "H2OVLChatModel", "HCXVisionV2ForCausalLM", "HF_ColBERT", "HYV3ForCausalLM",
    "HYV3MTPModel", "HYV4ForCausalLM", "HYV4MTPModel", "HfMoondream", "HrmTextForCausalLM",
    "HunYuanDenseV1ForCausalLM", "HunYuanMoEV1ForCausalLM", "HunYuanVLForConditionalGeneration",
    "HyperCLOVAXForCausalLM", "IQuestLoopCoderForCausalLM", "Idefics3ForConditionalGeneration",
    "InklingForCausalLM", "InklingForConditionalGeneration", "InklingMTPModel",
    "InternLM2ForCausalLM", "InternLM2ForRewardModel", "InternS1ForConditionalGeneration",
    "InternS1ProForConditionalGeneration", "InternS2MobiusForConditionalGeneration",
    "InternS2MobiusMTP", "InternS2PreviewForConditionalGeneration", "InternVLChatModel",
    "InternVLForConditionalGeneration", "IsaacForConditionalGeneration", "Jais2ForCausalLM",
    "JambaForCausalLM", "JambaForSequenceClassification", "JinaEmbeddingsV5Model", "JinaForRanking",
    "JinaVLForRanking", "K2HorizonForCausalLM", "K3DSparkModel", "KananaVForConditionalGeneration",
    "KeyeForConditionalGeneration", "KeyeVL1_5ForConditionalGeneration",
    "KimiK25ForConditionalGeneration", "KimiK3ForConditionalGeneration", "KimiK3MTPModel",
    "KimiLinearForCausalLM", "KimiVLForConditionalGeneration", "LagunaForCausalLM",
    "Lfm2ForCausalLM", "Lfm2MoeForCausalLM", "Lfm2VlForConditionalGeneration", "LiLiCorrDraftModel",
    "LightOnOCRForConditionalGeneration", "Llama4ForCausalLM", "Llama4ForConditionalGeneration",
    "LlamaBidirectionalForSequenceClassification", "LlamaBidirectionalModel",
    "LlamaForCausalLMEagle3", "LlamaNemotronVLForSequenceClassification", "LlamaNemotronVLModel",
    "Llama_Nemotron_Nano_VL", "LlavaForConditionalGeneration", "LlavaNextForConditionalGeneration",
    "LlavaNextVideoForConditionalGeneration", "LlavaOnevision2ForConditionalGeneration",
    "LlavaOnevisionForConditionalGeneration", "LongCatFlashMTPModel", "LongcatFlashForCausalLM",
    "LongcatFlashNgramForCausalLM", "Mamba2ForCausalLM", "MambaForCausalLM", "MellumForCausalLM",
    "MiDashengLMModel", "MiMoForCausalLM", "MiMoMTPModel", "MiMoV2FlashForCausalLM",
    "MiMoV2ForCausalLM", "MiMoV2MTPModel", "MiMoV2OmniForCausalLM", "MiMoV2OmniMTPModel",
    "MiniCPM3ForCausalLM", "MiniCPMForCausalLM", "MiniCPMO", "MiniCPMV",
    "MiniCPMV4_6ForConditionalGeneration", "MiniMaxM2ForCausalLM", "MiniMaxM3MTP",
    "MiniMaxM3SparseForCausalLM", "MiniMaxM3SparseForConditionalGeneration",
    "Mistral3ForConditionalGeneration", "MistralLarge3ForCausalLM", "MixtralForCausalLM",
    "ModernBertForSequenceClassification", "ModernBertForTokenClassification", "ModernBertModel",
    "Molmo2ForConditionalGeneration", "MolmoForCausalLM", "Moondream3ForCausalLM",
    "MoonshotKimiaForCausalLM", "MossAudioModel", "MossTranscribeDiarizeForConditionalGeneration",
    "MuseGlimmerAssistantModel", "MuseGlimmerForCausalLM", "MuseGlimmerForConditionalGeneration",
    "NVLM_D", "NanbeigeForCausalLM", "NemotronForCausalLM", "NemotronHMTPModel",
    "NemotronH_Nano_Omni_Reasoning_V3", "NemotronH_Nano_VL_V2", "NemotronH_Omni_Reasoning_V3",
    "NemotronH_Super_Omni_Reasoning_V3", "NemotronParseForConditionalGeneration", "NomicBertModel",
    "OPTForCausalLM", "Olmo2ForCausalLM", "Olmo3ForCausalLM", "OlmoForCausalLM",
    "OlmoHybridForCausalLM", "OlmoeForCausalLM", "OpenAIPrivacyFilterForTokenClassification",
    "OpenCUAForConditionalGeneration", "OpenPanguMTPModel", "OpenPanguVLForConditionalGeneration",
    "OpenVLAForActionPrediction", "OpsColQwen3Model", "OrionForCausalLM", "Ovis", "Ovis2_5",
    "Ovis2_6ForCausalLM", "Ovis2_6_MoeForCausalLM", "PEagleDraftModel",
    "PaddleOCRVLForConditionalGeneration", "PaliGemmaForConditionalGeneration",
    "PanguEmbeddedForCausalLM", "PanguProMoEV2ForCausalLM", "PanguUltraMoEForCausalLM",
    "Param2MoEForCausalLM", "PeagleLlamaForCausalLM", "PeagleQwen3ForCausalLM", "Phi3ForCausalLM",
    "Phi3VForCausalLM", "Phi4ForCausalLMV", "Phi4MMForCausalLM", "PhiForCausalLM",
    "PhiMoEForCausalLM", "PixtralForConditionalGeneration", "Plamo3ForCausalLM",
    "QianfanOCRForConditionalGeneration", "Qwen2AudioForConditionalGeneration",
    "Qwen2ForProcessRewardModel", "Qwen2ForRewardModel", "Qwen2MoeForCausalLM",
    "Qwen2_5OmniForConditionalGeneration", "Qwen2_5OmniModel", "Qwen3ASRForConditionalGeneration",
    "Qwen3ASRForcedAlignerForTokenClassification", "Qwen3ASRRealtimeGeneration", "Qwen3DSparkModel",
    "Qwen3MoeForCausalLM", "Qwen3NextForCausalLM", "Qwen3NextMTP", "Qwen3OmniDSparkModel",
    "Qwen3OmniMoeForConditionalGeneration", "Qwen3VLForConditionalGeneration",
    "Qwen3VLMoeForConditionalGeneration", "Qwen3VLNemotronEmbedModel", "Qwen3_5ForCausalLM",
    "Qwen3_5ForConditionalGeneration", "Qwen3_5MTP", "Qwen3_5MoeForCausalLM",
    "Qwen3_5MoeForConditionalGeneration", "Qwen3_5MoeMTP", "Qwen4ExpForCausalLM",
    "Qwen4ExpForConditionalGeneration", "Qwen4ExpMTP", "RForConditionalGeneration",
    "Rnj1ForCausalLM", "RobertaForMaskedLM", "RobertaForSequenceClassification",
    "RobertaForTokenClassification", "RobertaModel", "SarvamMLAForCausalLM", "SarvamMoEForCausalLM",
    "SeedOssForCausalLM", "SiglipModel", "SkyworkR1VChatModel", "SmolLM3ForCausalLM",
    "SmolVLMForConditionalGeneration", "SolarForCausalLM", "StableLmForCausalLM",
    "Starcoder2ForCausalLM", "Step1ForCausalLM", "Step3TextForCausalLM",
    "Step3VLForConditionalGeneration", "Step3p5ForCausalLM", "Step3p5MTP",
    "Step3p7ForConditionalGeneration", "StepVLForConditionalGeneration", "TeleChat2ForCausalLM",
    "TeleFLMForCausalLM", "Terratorch", "TransformersEmbeddingModel", "TransformersForCausalLM",
    "TransformersForSequenceClassification", "TransformersMoEEmbeddingModel",
    "TransformersMoEForCausalLM", "TransformersMoEForSequenceClassification",
    "TransformersMultiModalEmbeddingModel", "TransformersMultiModalForCausalLM",
    "TransformersMultiModalForSequenceClassification", "TransformersMultiModalMoEForCausalLM",
    "UltravoxModel", "UnlimitedOCRForCausalLM", "VaultGemmaForCausalLM",
    "VibeVoiceAsrForConditionalGeneration", "VoxtralForConditionalGeneration",
    "VoxtralRealtimeGeneration", "VoyageQwen3BidirectionalEmbedModel",
    "WhisperForConditionalGeneration", "XLMRobertaForSequenceClassification",
    "XLMRobertaForTokenClassification", "XLMRobertaModel", "Zamba2ForCausalLM",
];

pub(super) fn lookup(name: &str, minor: u8) -> Option<Registration> {
    match name {
        "CwmForCausalLM" => match minor {
            31 => Some(Registration::Decoder(Decoder::Llama)),
            32 | 33 => Some(Registration::Other),
            _ => None,
        },
        "FalconForCausalLM" => Some(Registration::Decoder(Decoder::Falcon)),
        "MedusaModel" if matches!(minor, 31 | 32) => Some(Registration::Other),
        "Qwen3ForCausalLM" => Some(Registration::Decoder(Decoder::Qwen3)),
        "Ministral3ForCausalLM" | "MistralForCausalLM" => {
            Some(Registration::Decoder(Decoder::Mistral))
        }
        "NemotronHForCausalLM" | "NemotronHPuzzleForCausalLM" => {
            Some(Registration::Decoder(Decoder::NemotronH))
        }
        "ClefOmniForJointSchema"
        | "EmbeddingGemma2Model"
        | "ExaoneMoEForCausalLM"
        | "MiniCPMV4_7ForConditionalGeneration"
            if matches!(minor, 32 | 33) =>
        {
            Some(Registration::Other)
        }
        "Qwen2ForCausalLM" | "Qwen2Model" => Some(Registration::Decoder(Decoder::Qwen2)),
        "Qwen2VLForConditionalGeneration" | "Qwen2_5_VLForConditionalGeneration" => {
            Some(Registration::Qwen2Vl)
        }
        "IQuestCoderForCausalLM"
        | "InternLM3ForCausalLM"
        | "LLaMAForCausalLM"
        | "LlamaForCausalLM"
        | "LlamaModel"
        | "MistralModel"
        | "TeleChat3ForCausalLM" => Some(Registration::Decoder(Decoder::Llama)),
        "AliceAIForCausalLM"
        | "Alpamayo2Super"
        | "BeamForCausalLM"
        | "ClefFlashForJointSchema"
        | "ColUltraVBertForRetrieval"
        | "D3ForSequenceClassification"
        | "DFlash2BeamDraftModel"
        | "DFlashQwen3XPressModel"
        | "DomynEdgeForCausalLM"
        | "DoryForCausalLM"
        | "FlintForCausalLM"
        | "Kolibri1ForCausalLM"
        | "Lfm2BidirectionalModel"
        | "MiniCPMOcrForConditionalGeneration"
        | "ModernVBertModel"
        | "MuseSparkDSparkModel"
        | "MuseSparkForConditionalGeneration"
        | "ParseAuxMTPModel"
        | "Qwen3XPressModel"
        | "RadioMBartPrefixForCausalLM"
        | "Step5ForCausalLM"
        | "Step5ForConditionalGeneration"
        | "Step5MTP"
        | "TransformersForMTP"
        | "TransformersMoEForMTP"
            if minor == 33 =>
        {
            Some(Registration::Other)
        }
        name if OTHER.binary_search(&name).is_ok() => Some(Registration::Other),
        _ => None,
    }
}
