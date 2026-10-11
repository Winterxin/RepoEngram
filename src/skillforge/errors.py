"""Errors callers can handle without parsing exception messages."""


class SkillForgeError(Exception):
    """Base error for expected SkillForge failures."""


class InvalidRepositoryPath(SkillForgeError):
    """The requested path is missing or is not a directory."""


class RepositoryReadError(SkillForgeError):
    """A repository could not be inspected reliably."""


class GitHistoryError(RepositoryReadError):
    """Git history could not be read or parsed reliably."""


class SemanticDecisionValidationError(ValueError):
    """A structured model decision failed deterministic semantic validation."""


class SemanticResponseValidationError(ValueError):
    """A model response failed validation without qualifying for a decision retry."""
