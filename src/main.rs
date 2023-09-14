mod neural_net;
use neural_net::{Network, WeightType};

fn main() {

	let mut net = neural_net::Network::new(neural_net::WeightType::Loaded(String::from("weights")));

	net.train(10, "data/mnist_train2.csv");

	//let mut net = Network::new(WeightType::Loaded(String::from("weights")));
	//net.train(10, "data/mnist_train2.csv");
	//net.test("test_data/test.csv", (1, 785));
}