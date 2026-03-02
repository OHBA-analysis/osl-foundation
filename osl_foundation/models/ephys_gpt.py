from osl_foundation.models.base import BaseModel
from osl_foundation.models.ephys_gpt_build import EphysGPTBuildMixin
from osl_foundation.models.ephys_gpt_sampling import EphysGPTSamplingMixin


class EphysGPT(EphysGPTBuildMixin, EphysGPTSamplingMixin, BaseModel):
    """
    EphysGPT model for electrophysiology data.

    Parameters
    ----------
    config : Config
        Config object.
    """
