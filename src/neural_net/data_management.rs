use csv::{ReaderBuilder, WriterBuilder};
use ndarray_csv::{Array2Reader, Array2Writer};
use std::fs::File;
use ndarray::*;

// returns first the training data as array then the testing data
pub fn load_data(filename: String, shape: (usize, usize)) -> Array2<f32> {
	let file = File::open(filename).unwrap();
	let mut reader = ReaderBuilder::new().has_headers(false).from_reader(file);

	return reader.deserialize_array2(shape).unwrap();
}

pub fn save_to_file(arr: &Array2<f32>, filename: &str) {
	let file = File::create(filename).unwrap();
	let mut writer = WriterBuilder::new().has_headers(false).from_writer(file);
	writer.serialize_array2(arr).unwrap();
}