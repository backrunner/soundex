//! Offline interventions only. No production config or model contract accepts these.

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FusionAblation {
    #[default]
    Current,
    MagnitudeOnly,
    PhaseOnly,
    Unfused,
    SafetyOnly,
}

impl FusionAblation {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Current => "current",
            Self::MagnitudeOnly => "magnitude-only",
            Self::PhaseOnly => "phase-only",
            Self::Unfused => "unfused",
            Self::SafetyOnly => "safety-only",
        }
    }
}

impl std::str::FromStr for FusionAblation {
    type Err = &'static str;
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "current" => Ok(Self::Current),
            "magnitude-only" => Ok(Self::MagnitudeOnly),
            "phase-only" => Ok(Self::PhaseOnly),
            "unfused" => Ok(Self::Unfused),
            "safety-only" => Ok(Self::SafetyOnly),
            _ => Err("unknown diagnostic fusion ablation"),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ConditionAblation {
    #[default]
    Native,
    FixedCutoff,
    ActiveOne,
    FixedDetector,
}

impl ConditionAblation {
    pub fn apply(self, condition: &mut [f32]) {
        // Fixed predeclared interventions, not estimated from the reference or clip.
        // Inference sensitivity of frozen weights is NOT a no-condition training trial.
        if matches!(self, Self::FixedCutoff | Self::FixedDetector) {
            condition[1] = 0.5;
        }
        if matches!(self, Self::ActiveOne | Self::FixedDetector) {
            condition[2] = 1.0;
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Native => "native",
            Self::FixedCutoff => "fixed-cutoff",
            Self::ActiveOne => "active-one",
            Self::FixedDetector => "fixed-detector",
        }
    }
}

impl std::str::FromStr for ConditionAblation {
    type Err = &'static str;
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "native" => Ok(Self::Native),
            "fixed-cutoff" => Ok(Self::FixedCutoff),
            "active-one" => Ok(Self::ActiveOne),
            "fixed-detector" => Ok(Self::FixedDetector),
            _ => Err("unknown diagnostic condition ablation"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn conditions_preserve_rate_and_only_intervene_on_requested_inputs() {
        for (mode, expected) in [
            (ConditionAblation::Native, [0.91875, 0.8, 0.0]),
            (ConditionAblation::FixedCutoff, [0.91875, 0.5, 0.0]),
            (ConditionAblation::ActiveOne, [0.91875, 0.8, 1.0]),
            (ConditionAblation::FixedDetector, [0.91875, 0.5, 1.0]),
        ] {
            let mut condition = [0.91875, 0.8, 0.0];
            mode.apply(&mut condition);
            assert_eq!(condition, expected);
            assert_eq!(mode.as_str().parse(), Ok(mode));
        }
        assert!("typo".parse::<ConditionAblation>().is_err());
        assert!("typo".parse::<FusionAblation>().is_err());
    }
}
