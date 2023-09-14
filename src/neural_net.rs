mod data_management;
pub mod network_functions;

use data_management::*;
use network_functions::*;

use ndarray::*;
use ndarray_rand::{*, rand_distr::Uniform};




pub enum WeightType {
	Random,
	Loaded(String),
}

pub struct Network {

	weights_inp_hid: Array2<f32>,
	weights_hid_out: Array2<f32>,

	bias_inp_hid: Array2<f32>,
	bias_hid_out: Array2<f32>,

	learn_rate: f32,
}

impl Network {
	pub fn new(weights: WeightType) -> Network {

		let weights_inp_hid:Array2<f32>;
		let weights_hid_out:Array2<f32>;

		let bias_inp_hid: Array2<f32>;
		let bias_hid_out: Array2<f32>;

		match weights {
			WeightType::Random => {
				weights_inp_hid = Array::random((20, 784), Uniform::new(-0.5, 0.5));
				weights_hid_out = Array::random((10, 20), Uniform::new(-0.5, 0.5));
			
				bias_inp_hid = ndarray::Array2::zeros((20,1));
				bias_hid_out = ndarray::Array2::zeros((10,1));
			}
			WeightType::Loaded(folder) => {
				weights_inp_hid = load_data(folder.to_owned() + "/weights_inp_hid.csv", (20, 784));
				weights_hid_out = load_data(folder.to_owned() + "/weights_hid_out.csv", (10, 20));
			
				bias_inp_hid = load_data(folder.to_owned() + "/bias_inp_hid.csv", (20, 1));
				bias_hid_out = load_data(folder.to_owned() + "/bias_hid_out.csv", (10, 1));
			}
		}

		let learn_rate: f32 = 0.01;

		Network {weights_inp_hid, weights_hid_out, bias_inp_hid, bias_hid_out, learn_rate}
	}


	fn feedforward(&self, i: Array1<f32>) -> (Array2<f32>, Array2<f32>, Array2<f32>) {
		let inp: Array1<f32> = i.slice(s![1..]).to_owned();
		let inp: Array2<f32> = inp.into_shape((784, 1)).unwrap();

		// input -> hidden
		let hid_pre = self.weights_inp_hid.dot(&inp) + &self.bias_inp_hid;

		let mut hid = hid_pre.to_owned();
		hid.mapv_inplace(|x| sigmoid(x));

		// hidden -> output
		let out_pre = self.weights_hid_out.dot(&hid) + &self.bias_hid_out;
		let mut out: Array2<f32> = out_pre.to_owned();
		out.mapv_inplace(|x| sigmoid(x));


		(out, hid, inp)

	}

	pub fn train(&mut self, epochs: usize, filename: &str) {

		let data: Array2<f32> = load_data(String::from(filename), (60000, 785));

		let mut highest_acc: f32 = 0.0;

		for ep in 0..epochs {
			println!("----------");
			println!("beginning epoch: {}", ep);

			for i in data.rows() {

				let label: Array2<f32> = one_hot(i[0] as usize);

				let feed_forward_out = self.feedforward(i.to_owned());


				// backprop
				let out = feed_forward_out.0;
				let hid = feed_forward_out.1;
				let inp = feed_forward_out.2;

				// backprop from out to hidden layers
				let delta_out: Array2<f32> = &out - &label;
				self.weights_hid_out = self.weights_hid_out.to_owned() + ((-self.learn_rate * &delta_out).dot(&hid.t()));
				self.bias_hid_out = self.bias_hid_out.to_owned() + (-self.learn_rate * &delta_out);
	
				// backprop from hidden to input layers
				let delta_hid: Array2<f32> = self.weights_hid_out.t().dot(&delta_out) * (&hid * (hid.mapv(|x| 1.0-x)));
				self.weights_inp_hid = self.weights_inp_hid.to_owned() + (-self.learn_rate * delta_hid.dot(&inp.t()));
				self.bias_inp_hid = self.bias_inp_hid.to_owned() + (-self.learn_rate * delta_hid);
			}
			println!("epoch: {} completed", ep);
			let acc: f32 = self.test("data/mnist_test2.csv", (10000, 785));

			if acc > highest_acc {
				highest_acc = acc;
				println!("new highest accuracy achieved, saving weights");

				save_to_file(&self.weights_inp_hid, "weights/weights_inp_hid.csv");
				save_to_file(&self.weights_hid_out, "weights/weights_hid_out.csv");
		
				save_to_file(&self.bias_inp_hid, "weights/bias_inp_hid.csv");
				save_to_file(&self.bias_hid_out, "weights/bias_hid_out.csv");

				println!("finished saving");


			}
		}
	}

	pub fn test(&self, filename: &str, shape: (usize, usize)) -> f32 {

		let mut num_correct: u32 = 0;

		let data: Array2<f32> = load_data(String::from(filename), shape);

		for i in data.rows() {
			let label: Array2<f32> = one_hot(i[0] as usize);
			let out = self.feedforward(i.to_owned()).0;

			//println!("{:?}", out);
			//println!("{}", de_one_hot(&out));

			if de_one_hot(&label) == de_one_hot(&out) {
				num_correct += 1;
			}

			println!("{}", de_one_hot(&out));


		}

		let acc = round_to_place((num_correct as f32 / 10000.0)*100.0, 2);

		println!("testing finished, num correct: {}", num_correct);

		acc

	}


}