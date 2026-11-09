// Householder QR avoids forming the normal equations, which square the condition number.
pub fn least_squares(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = a.first()?.len();
    if a.len() < n {
        return None;
    }
    for k in 0..n {
        let mut reflector: Vec<_> = a[k..].iter().map(|row| row[k]).collect();
        let norm = reflector.iter().map(|v| v * v).sum::<f64>().sqrt();
        if !norm.is_finite() || norm < 1e-13 {
            return None;
        }
        reflector[0] += norm.copysign(reflector[0]);
        let factor = 2.0 / reflector.iter().map(|v| v * v).sum::<f64>();
        for j in k..n {
            let dot = factor
                * a[k..]
                    .iter()
                    .zip(&reflector)
                    .map(|(row, v)| row[j] * v)
                    .sum::<f64>();
            for (row, v) in a[k..].iter_mut().zip(&reflector) {
                row[j] -= dot * v;
            }
        }
        let dot = factor * dot(&b[k..], &reflector);
        for (y, v) in b[k..].iter_mut().zip(&reflector) {
            *y -= dot * v;
        }
    }
    let mut weights = vec![0.0; n];
    for k in (0..n).rev() {
        weights[k] = (b[k] - dot(&a[k][k + 1..], &weights[k + 1..])) / a[k][k];
    }
    weights.iter().all(|v| v.is_finite()).then_some(weights)
}

pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

pub fn cholesky(mut a: Vec<Vec<f64>>) -> Option<Vec<Vec<f64>>> {
    for i in 0..a.len() {
        for j in 0..=i {
            let value = a[i][j] - dot(&a[i][..j], &a[j][..j]);
            a[i][j] = if i == j {
                if !value.is_finite() || value <= 0.0 {
                    return None;
                }
                value.sqrt()
            } else {
                value / a[j][j]
            };
        }
    }
    Some(a)
}

pub fn forward(l: &[Vec<f64>], b: &mut [f64]) {
    for (i, row) in l.iter().enumerate() {
        b[i] = (b[i] - dot(&row[..i], &b[..i])) / row[i];
    }
}

pub fn backward(l: &[Vec<f64>], b: &mut [f64]) {
    for i in (0..l.len()).rev() {
        let sum: f64 = (i + 1..l.len()).map(|j| l[j][i] * b[j]).sum();
        b[i] = (b[i] - sum) / l[i][i];
    }
}
