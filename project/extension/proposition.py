from enum import Enum, auto

class Proposition(Enum):
    REACH_ZONE_A = auto()
    REACH_ZONE_B = auto()
    AVOID_ZONE_A = auto()
    AVOID_ZONE_B = auto()
    REACH_ZONE_C = auto()
    AVOID_ZONE_C = auto()
    WVF_MAX = auto()
    WVF_MIN = auto()

    @staticmethod
    def logical_negation(prop):
        match prop:
            case Proposition.REACH_ZONE_A:
                return Proposition.AVOID_ZONE_A
            case Proposition.AVOID_ZONE_A:
                return Proposition.REACH_ZONE_A
            case Proposition.REACH_ZONE_B:
                return Proposition.AVOID_ZONE_B
            case Proposition.AVOID_ZONE_B:
                return Proposition.REACH_ZONE_B
            case Proposition.AVOID_ZONE_C:
                return Proposition.REACH_ZONE_C
            case Proposition.REACH_ZONE_C:
                return Proposition.AVOID_ZONE_C
