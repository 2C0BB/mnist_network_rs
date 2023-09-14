use ndarray::*;

pub fn sigmoid(n: f32) -> f32 {
	1.0 / (1.0 + std::f32::consts::E.powf(-n))
}

pub fn one_hot(label: usize) -> Array2<f32> {
	let mut enc: Array2<f32> = Array2::zeros((10,1));
	enc.row_mut(label)[0] = 1.;

	enc
}

pub fn de_one_hot(a: &Array2<f32>) -> u32 {

	let mut max_idx: usize = 0;
	let mut prev_max: f32 = 0.;


	for (idx, i) in a.iter().enumerate() {

		if *i > prev_max {
			
			prev_max = *i;
			max_idx = idx;
		}
	}

	max_idx as u32
}

pub fn round_to_place(n:f32, place:u32) -> f32 {
	(n*((10 as f32).powf(place as f32))).round() / (10 as f32).powf(place as f32)
}