"""Deterministic specialist agents. Replace/augment rules with model providers as needed."""
from __future__ import annotations
from .contracts import EventContext, AgentResult


def _attendance(event: EventContext) -> int:
    # Prefer supplied forecast; otherwise use registration count as a conservative baseline.
    return min(event.venue_capacity, event.expected_attendance if event.expected_attendance is not None else event.registrations)


def event_planning_agent(event: EventContext, query: str = "") -> AgentResult:
    expected = _attendance(event)
    occupancy = expected / event.venue_capacity
    recommendations = []
    if event.registrations > event.venue_capacity:
        recommendations.append("Registrations exceed venue capacity; apply a waitlist or capacity cap before admitting attendees.")
    if occupancy >= 0.85:
        recommendations.append("Plan controlled entry, clear queue lanes, and a staffed overflow/holding area.")
    elif occupancy >= 0.65:
        recommendations.append("Monitor entry rates and keep circulation routes clear during peak arrival.")
    else:
        recommendations.append("Current forecast leaves capacity headroom; confirm the estimate against registration and historical event data.")
    if event.budget is None:
        recommendations.append("Set a budget and cost estimates for venue, staff, catering, security, and contingency.")
    if query.strip():
        recommendations.append(f"Planning request recorded for follow-up: {query.strip()[:240]}")
    return AgentResult(agent="event_planning", summary=f"Planning baseline for {event.event_name}: forecast {expected} attendees against capacity {event.venue_capacity}.", recommendations=recommendations, metrics={"expected_attendance": expected, "venue_capacity": event.venue_capacity, "forecast_occupancy": round(occupancy, 3), "registrations": event.registrations})


def staffing_agent(event: EventContext) -> AgentResult:
    expected = _attendance(event)
    # Transparent starter heuristics, not a certified safety standard.
    managers = max(1, (expected + 249) // 250)
    volunteers = max(1, (expected + 49) // 50)
    security = max(1, (expected + 99) // 100)
    total = managers + volunteers + security
    recs = [f"Baseline staffing estimate: {managers} manager(s), {volunteers} volunteer(s), {security} security staff."]
    warnings = ["Staffing ratios are prototype heuristics; validate against local law, venue policy, event risk, and a qualified safety lead."]
    if event.staff_available is not None and event.staff_available < total:
        recs.append(f"Shortfall of at least {total - event.staff_available} staff against this baseline; recruit or reduce the operating scope.")
    return AgentResult(agent="staffing", summary=f"Estimated {total} staff for forecast attendance of {expected}.", recommendations=recs, metrics={"expected_attendance": expected, "managers": managers, "volunteers": volunteers, "security": security, "total_recommended": total, "staff_available": event.staff_available}, warnings=warnings)


def crowd_risk_agent(event: EventContext) -> AgentResult:
    observed = event.current_attendance if event.current_attendance is not None else _attendance(event)
    occupancy = observed / event.venue_capacity
    if occupancy >= 0.95:
        level = "critical"
        recs = ["Immediately pause or meter entry and verify the count with on-site staff.", "Keep exits and emergency access routes clear; follow the venue incident plan."]
    elif occupancy >= 0.85:
        level = "high"
        recs = ["Issue a crowding alert and deploy staff to the affected zone.", "Consider one-way routing or redirecting arrivals to a verified lower-density area."]
    elif occupancy >= 0.65:
        level = "moderate"
        recs = ["Increase monitoring frequency and prepare crowd-redirection signage."]
    else:
        level = "low"
        recs = ["Continue routine monitoring; this estimate alone does not establish safe crowd conditions."]
    warnings = ["This is a capacity-ratio rule, not computer-vision analysis. Do not use it as the sole basis for emergency decisions."]
    return AgentResult(agent="crowd_risk", summary=f"Crowd-risk heuristic is {level} at {occupancy:.1%} of venue capacity.", recommendations=recs, metrics={"observed_or_estimated_attendance": observed, "venue_capacity": event.venue_capacity, "occupancy_ratio": round(occupancy, 3), "risk_level": level}, warnings=warnings)


def general_agent(event: EventContext, query: str = "") -> AgentResult:
    return AgentResult(agent="general", summary=f"Received request for event {event.event_name}.", recommendations=["Specify whether you need event planning, staffing/resource planning, or crowd-risk assessment.", *( [f"Request: {query.strip()[:240]}"] if query.strip() else [])], metrics={"event_id": event.event_id})
