use ml_lab_leptos::{
    data::{DataConfig, Dataset, Point, fold, grid, metrics, target},
    model::{Fit, Method, Model, Parameters},
};

fn near(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "{actual} differs from {expected}"
    );
}

#[test]
fn opening_example_matches_original_lab() {
    let config = DataConfig::default();
    let data = Dataset::generate(config);
    let fit = Fit::train(
        &data.train,
        Parameters::initial(Method::Polynomial),
        false,
        config.seed,
    )
    .unwrap();
    near(data.train[2].y, 2.2737419604058524, 1e-12);
    near(fit.predict(0.5), -0.5480884604981248, 1e-12);
    near(
        metrics(&data.train, |x| fit.predict(x)).rmse,
        1.0815071448793168,
        1e-12,
    );
    near(
        metrics(&data.test, |x| fit.predict(x)).rmse,
        0.824493697744832,
        1e-12,
    );
}

#[test]
fn all_seven_methods_are_finite_at_full_capacity() {
    let config = DataConfig {
        count: 200,
        ..DataConfig::default()
    };
    let data = Dataset::generate(config);
    for periodic in [false, true] {
        let points = if periodic {
            fold(&data.train)
        } else {
            data.train.clone()
        };
        for method in Method::ALL {
            for index in 0..3 {
                let params = Parameters::preset(method, index, points.len(), periodic);
                let fit = Fit::train(&points, params, periodic, config.seed).unwrap();
                for x in grid(-9.0, 9.0, 61) {
                    assert!(
                        fit.predict(x).is_finite(),
                        "{method:?}, preset {index}, x={x}"
                    );
                    assert!(
                        fit.uncertainty(x)
                            .is_none_or(|sd| sd.is_finite() && (0.0..=1.0).contains(&sd))
                    );
                }
            }
        }
    }
}

#[test]
fn tree_and_nearest_neighbor_can_interpolate_training_points() {
    let config = DataConfig::default();
    let data = Dataset::generate(config);
    for method in [Method::Tree, Method::Knn] {
        let fit = Fit::train(
            &data.train,
            Parameters::preset(method, 2, data.train.len(), false),
            false,
            config.seed,
        )
        .unwrap();
        for point in &data.train {
            near(fit.predict(point.x), point.y, 1e-12);
        }
    }
}

#[test]
fn outliers_leave_test_set_and_training_coordinates_intact() {
    let config = DataConfig::default();
    let corrupt = Dataset::generate(config);
    let clean = Dataset::generate(DataConfig {
        outliers: false,
        ..config
    });
    assert_eq!(corrupt.test, clean.test);
    let mut changed = 0;
    for (a, b) in corrupt.train.iter().zip(clean.train) {
        assert_eq!(a.x, b.x);
        if (a.y - b.y).abs() > 1e-9 {
            changed += 1;
        }
    }
    assert_eq!(changed, 3);
}

#[test]
fn periodic_input_repeats_predictions_and_uncertainty() {
    let data = Dataset::generate(DataConfig::default());
    let fit = Fit::train(
        &fold(&data.train),
        Parameters::initial(Method::Gp),
        true,
        72,
    )
    .unwrap();
    for x in [-0.55, 0.24, 0.88] {
        near(fit.predict(x), fit.predict(x + 8.0), 1e-10);
        near(
            fit.uncertainty(x).unwrap(),
            fit.uncertainty(x + 8.0).unwrap(),
            1e-10,
        );
    }
}

#[test]
fn adam_reduces_loss_and_reference_checkpoint_resumes() {
    let points: Vec<_> = grid(-1.0, 1.0, 30)
        .map(|x| Point { x, y: target(x) })
        .collect();
    let fit = Fit::train(&points, Parameters::initial(Method::Neural), false, 42).unwrap();
    let Model::Neural(mut network) = fit.model else {
        panic!("expected neural network")
    };
    let initial = metrics(&points, |x| network.predict(x)).rmse;
    for _ in 0..300 {
        assert!(network.step(&points));
    }
    assert!(metrics(&points, |x| network.predict(x)).rmse < initial * 0.5);
    assert!(network.restore_reference());
    assert_eq!(network.epoch, 6000);
    assert!(network.step(&points));
    assert_eq!(network.epoch, 6001);
}

#[test]
fn manual_marks_only_change_training_when_exclusion_is_enabled() {
    use ml_lab_leptos::lab::{Lab, index};
    let mut lab = Lab::new();
    let method = index(Method::Polynomial);
    let before = lab.scores[method].unwrap()[1].rmse;
    let test = lab.data.test.clone();
    lab.mark(2);
    assert_eq!(lab.marked.len(), 1);
    assert_eq!(lab.used_count(), 24);
    assert_eq!(lab.scores[method].unwrap()[1].rmse, before);
    lab.exclude = true;
    lab.refit_all();
    assert_eq!(lab.used_count(), 23);
    assert!(lab.scores[method].unwrap()[1].rmse < before);
    assert_eq!(lab.data.test, test);
    lab.mark(2);
    assert_eq!(lab.used_count(), 24);
    near(lab.scores[method].unwrap()[1].rmse, before, 1e-12);
    lab.mark(2);
    lab.regenerate();
    assert!(lab.marked.is_empty());
}

#[test]
fn marking_preserves_enough_phases_for_a_later_periodic_fit() {
    use ml_lab_leptos::lab::Lab;
    let mut lab = Lab::new();
    lab.data_config.count = 10;
    lab.regenerate();
    for i in 0..lab.data.train.len() {
        lab.mark(i);
    }
    lab.exclude = true;
    lab.periodic = true;
    lab.refit_all();
    assert!(lab.training.len() >= 2);
    assert!(lab.models.iter().all(Result::is_ok));
}
