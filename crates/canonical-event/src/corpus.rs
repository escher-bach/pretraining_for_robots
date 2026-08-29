//! A small reference corpus of canonical episodes.
//!
//! The corpus is hand-written rather than generated, so that it covers every
//! representable fact kind and every profile disagreement on purpose. The
//! conformance test that reads the real diagnostic crate's output is what checks
//! agreement with production; this corpus is what checks coverage.

use crate::{
    ActionTarget, ChannelContent, ChannelRole, Episode, EventGroup, KeyNamespace, LocalKey,
    Profile, PublicEpisode, PublicFact, PublicRecord, Quantity, QueryHorizon, RecordAddress,
    StepSpan, SupervisionRecord, SupervisionTable, ACTION_HORIZON, DIAGNOSTIC_CONTROL_HORIZON,
};

/// Five positions on a line, matching the finite diagnostic.
pub const LINE_LENGTH: u16 = 5;
/// The diagnostic's single continuous movement control.
pub const DIAGNOSTIC_CONTROL_KEY: u16 = 30;
/// The monomial corpus instance has two observation and two actuator channels.
pub const MONOMIAL_DIMENSION: u16 = 2;
/// The monomial corpus instance's per-step action limit, in physical units.
pub const MONOMIAL_ACTION_LIMIT: f64 = 0.25;

fn observation_key(name: u16) -> LocalKey {
    LocalKey::new(KeyNamespace::Observation, name)
}

fn actuator_key(name: u16) -> LocalKey {
    LocalKey::new(KeyNamespace::Actuator, name)
}

fn episode_key(name: u16) -> LocalKey {
    LocalKey::new(KeyNamespace::Episode, name)
}

fn record(key: LocalKey, fact: PublicFact) -> PublicRecord {
    PublicRecord::new(key, fact).expect("corpus records are well typed")
}

/// The line coordinate of a position, normalized to `[-1, 1]`.
///
/// Every value is a dyadic rational, so it survives the float layout exactly.
pub fn normalized_position(coordinate: u16) -> f64 {
    f64::from(coordinate) / f64::from(LINE_LENGTH - 1) * 2.0 - 1.0
}

fn selection() -> ChannelContent {
    ChannelContent::Selection {
        indicator: Quantity::normalized(1.0, 0.0, 1.0),
    }
}

fn diagnostic_span(remaining: u16) -> StepSpan {
    StepSpan::new(remaining, DIAGNOSTIC_CONTROL_HORIZON).expect("remaining steps fit the horizon")
}

fn monomial_span() -> StepSpan {
    StepSpan::new(1, ACTION_HORIZON as u16).expect("one step fits the action head")
}

/// One episode of the finite goal diagnostic, in canonical form.
///
/// `goal_position` is the requested end position and `displacement` is the move
/// the teacher supervises at each of the two steps. Only the goal record and the
/// supervised displacement differ between the two witnesses, which is what makes
/// the goal-sensitivity test sharp.
pub fn diagnostic_episode(goal_position: u16, displacement: f64) -> Episode {
    let start = 2u16;
    let mut groups = Vec::new();
    let mut supervision = SupervisionTable::default();

    let mut schema: Vec<PublicRecord> = (0..LINE_LENGTH)
        .map(|coordinate| {
            record(
                observation_key(coordinate),
                PublicFact::ChannelSchema {
                    channel: ChannelRole::Observation,
                    reference: Quantity::normalized(normalized_position(coordinate), -1.0, 1.0),
                    command_span: None,
                },
            )
        })
        .collect();
    schema.push(record(
        actuator_key(DIAGNOSTIC_CONTROL_KEY),
        PublicFact::ChannelSchema {
            channel: ChannelRole::Actuator,
            reference: Quantity::normalized(0.0, -1.0, 1.0),
            command_span: None,
        },
    ));
    groups.push(EventGroup {
        group: 0,
        records: schema,
    });

    groups.push(EventGroup {
        group: 1,
        records: vec![record(
            episode_key(0),
            PublicFact::Boundary {
                subtype: crate::BoundarySubtype::TaskReset,
            },
        )],
    });
    groups.push(EventGroup {
        group: 2,
        records: vec![record(
            observation_key(goal_position),
            PublicFact::Goal {
                content: selection(),
            },
        )],
    });
    groups.push(EventGroup {
        group: 3,
        records: vec![record(
            observation_key(start),
            PublicFact::Observation {
                content: selection(),
            },
        )],
    });

    let mut position = start;
    let mut group_index = 4u16;
    for step in 0..DIAGNOSTIC_CONTROL_HORIZON {
        groups.push(EventGroup {
            group: group_index,
            records: vec![record(
                actuator_key(DIAGNOSTIC_CONTROL_KEY),
                PublicFact::ActionQuery {
                    command: Quantity::normalized(0.0, -1.0, 1.0),
                    horizon: QueryHorizon::RemainingFraction {
                        remaining: diagnostic_span(DIAGNOSTIC_CONTROL_HORIZON - step),
                    },
                },
            )],
        });
        supervision.set(
            RecordAddress {
                group_index: groups.len() - 1,
                record_index: 0,
            },
            SupervisionRecord {
                action_targets: vec![ActionTarget {
                    step: 0,
                    value: displacement,
                }],
                future_target: None,
            },
        );
        group_index += 1;

        groups.push(EventGroup {
            group: group_index,
            records: vec![record(
                actuator_key(DIAGNOSTIC_CONTROL_KEY),
                PublicFact::ActionExecuted {
                    command: Quantity::normalized(displacement, -1.0, 1.0),
                    actuator_marker: true,
                },
            )],
        });
        group_index += 1;

        position = position.saturating_add_signed(displacement as i16);
        groups.push(EventGroup {
            group: group_index,
            records: vec![record(
                observation_key(position),
                PublicFact::Observation {
                    content: selection(),
                },
            )],
        });
        group_index += 1;
    }

    groups.push(EventGroup {
        group: group_index,
        records: vec![record(
            episode_key(0),
            PublicFact::Boundary {
                subtype: crate::BoundarySubtype::EpisodeEnd,
            },
        )],
    });

    Episode::new(
        PublicEpisode::new(Profile::GoalConditionedDiagnostic, groups),
        supervision,
    )
    .expect("corpus supervision addresses existing records")
}

/// One calibration-and-control episode in the monomial profile.
///
/// This covers the three kinds the diagnostic never emits: future queries,
/// feedback, and the calibration boundary subtype.
pub fn monomial_episode() -> Episode {
    let mut groups = Vec::new();
    let mut supervision = SupervisionTable::default();
    let limit = MONOMIAL_ACTION_LIMIT;

    let mut schema = Vec::new();
    for channel in 0..MONOMIAL_DIMENSION {
        schema.push(record(
            observation_key(channel),
            PublicFact::ChannelSchema {
                channel: ChannelRole::Observation,
                reference: Quantity::normalized(0.0, -1.0, 1.0),
                command_span: None,
            },
        ));
    }
    for channel in 0..MONOMIAL_DIMENSION {
        schema.push(record(
            actuator_key(channel),
            PublicFact::ChannelSchema {
                channel: ChannelRole::Actuator,
                reference: Quantity::normalized(0.0, -limit, limit),
                command_span: Some(monomial_span()),
            },
        ));
    }
    groups.push(EventGroup {
        group: 0,
        records: schema,
    });

    groups.push(EventGroup {
        group: 1,
        records: vec![record(
            episode_key(0),
            PublicFact::Boundary {
                subtype: crate::BoundarySubtype::CalibrationReset,
            },
        )],
    });
    groups.push(EventGroup {
        group: 2,
        records: observation_values(&[0.0, 0.0]),
    });
    groups.push(EventGroup {
        group: 3,
        records: vec![
            record(
                actuator_key(0),
                PublicFact::ActionExecuted {
                    command: Quantity::normalized(0.5, -limit, limit),
                    actuator_marker: true,
                },
            ),
            record(
                actuator_key(1),
                PublicFact::ActionExecuted {
                    command: Quantity::normalized(0.0, -limit, limit),
                    actuator_marker: true,
                },
            ),
        ],
    });
    groups.push(EventGroup {
        group: 4,
        records: (0..MONOMIAL_DIMENSION)
            .map(|channel| {
                record(
                    observation_key(channel),
                    PublicFact::FutureQuery {
                        command: Quantity::normalized(0.0, -1.0, 1.0),
                        horizon: monomial_span(),
                    },
                )
            })
            .collect(),
    });
    for channel in 0..MONOMIAL_DIMENSION {
        supervision.set(
            RecordAddress {
                group_index: 4,
                record_index: channel as usize,
            },
            SupervisionRecord {
                action_targets: Vec::new(),
                future_target: Some(if channel == 0 { 0.0 } else { 0.125 }),
            },
        );
    }
    groups.push(EventGroup {
        group: 5,
        records: observation_values(&[0.0, 0.125]),
    });
    groups.push(EventGroup {
        group: 6,
        records: vec![record(
            episode_key(0),
            PublicFact::Boundary {
                subtype: crate::BoundarySubtype::TaskReset,
            },
        )],
    });
    groups.push(EventGroup {
        group: 7,
        records: vec![
            record(
                observation_key(0),
                PublicFact::Goal {
                    content: ChannelContent::Value(Quantity::normalized(0.5, -1.0, 1.0)),
                },
            ),
            record(
                observation_key(1),
                PublicFact::Goal {
                    content: ChannelContent::Value(Quantity::normalized(-0.25, -1.0, 1.0)),
                },
            ),
        ],
    });
    groups.push(EventGroup {
        group: 8,
        records: observation_values(&[0.0, 0.0]),
    });
    groups.push(EventGroup {
        group: 9,
        records: (0..MONOMIAL_DIMENSION)
            .map(|channel| {
                record(
                    actuator_key(channel),
                    PublicFact::ActionQuery {
                        command: Quantity::normalized(0.0, -limit, limit),
                        horizon: QueryHorizon::ActuatorSpan {
                            marker: true,
                            requested: monomial_span(),
                        },
                    },
                )
            })
            .collect(),
    });
    for channel in 0..MONOMIAL_DIMENSION {
        supervision.set(
            RecordAddress {
                group_index: 9,
                record_index: channel as usize,
            },
            SupervisionRecord {
                action_targets: vec![ActionTarget {
                    step: 0,
                    value: if channel == 0 { 1.0 } else { -0.5 },
                }],
                future_target: None,
            },
        );
    }
    groups.push(EventGroup {
        group: 10,
        records: vec![
            record(
                actuator_key(0),
                PublicFact::ActionExecuted {
                    command: Quantity::normalized(1.0, -limit, limit),
                    actuator_marker: true,
                },
            ),
            record(
                actuator_key(1),
                PublicFact::ActionExecuted {
                    command: Quantity::normalized(-0.5, -limit, limit),
                    actuator_marker: true,
                },
            ),
        ],
    });
    groups.push(EventGroup {
        group: 11,
        records: observation_values(&[0.25, -0.125]),
    });
    groups.push(EventGroup {
        group: 12,
        records: vec![record(
            episode_key(0),
            PublicFact::Feedback {
                error: Quantity::normalized(0.125, 0.0, 1.0),
                success: false,
            },
        )],
    });
    groups.push(EventGroup {
        group: 13,
        records: vec![record(
            episode_key(0),
            PublicFact::Boundary {
                subtype: crate::BoundarySubtype::EpisodeEnd,
            },
        )],
    });

    Episode::new(
        PublicEpisode::new(Profile::CalibratedMonomial, groups),
        supervision,
    )
    .expect("corpus supervision addresses existing records")
}

fn observation_values(values: &[f64]) -> Vec<PublicRecord> {
    values
        .iter()
        .enumerate()
        .map(|(channel, value)| {
            record(
                observation_key(channel as u16),
                PublicFact::Observation {
                    content: ChannelContent::Value(Quantity::normalized(*value, -1.0, 1.0)),
                },
            )
        })
        .collect()
}

/// The named reference corpus.
pub fn reference_episodes() -> Vec<(&'static str, Episode)> {
    vec![
        ("diagnostic_goal_left", diagnostic_episode(0, -1.0)),
        ("diagnostic_goal_right", diagnostic_episode(4, 1.0)),
        ("monomial_calibrate_and_control", monomial_episode()),
    ]
}
