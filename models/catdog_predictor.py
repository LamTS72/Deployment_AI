import sys
import torch
import torchvision

from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from PIL import Image
from torch.nn import functional as F
from utils.logger import Logger
from config.catdog_cfg import CatDogDataConfig, ModelConfig
from .catdog_model import CatDogModel
from torchvision import transforms

LOGGER = Logger(__file__, log_file="predictor.log")
LOGGER.log.info("Starting Model Serving")


class Predictor():
	def __init__(self, model_name, model_weights, device="cpu"):
			self.model_name = model_name
			self.model_weights = model_weights
			self.model = self.load_model()
			self.self_transforms = self.create_transform()

	async def predict(self, image):
		pil_img = Image.open(image)
		if pil_img.mode == "RGBA":
			pil_img = pil_img.convert("RGB")
		transformed_image = self.transforms_(pil_img).unsqueeze(0)
		output = await self.model_inference(transformed_image)
		probs, best_prob, predicted_id, predicted_class = self.output2pred(output)
		print(output)
		LOGGER.log_model(self.model_name)
		LOGGER.log_response(best_prob, predicted_id, predicted_class)
		torch.cuda.empty_cache()

		resp_dict  = {
				"probs": probs,
				"best_prob": best_prob,
				"predicted_id": predicted_id,
				"predicted_class": predicted_class,
				"predictor_name": self.model_name
		}
		return resp_dict
	
	async def model_inference(self, transformed_image):
		self.model.eval()
		with torch.inference_mode():
			output = self.model(transformed_image)
		return output

	def output2pred(self, output):
			probs = F.softmax(output, dim=1)
			best_prob =  torch.max(probs, dim=1)[0].item()
			predicted_id =  torch.max(probs, dim=1)[1].item()
			predicted_class = CatDogDataConfig.ID2LABEL[predicted_id]
			return probs.squeeze().tolist(), best_prob, predicted_id, predicted_class

	def load_model(self):
		try:
			model = CatDogModel(CatDogDataConfig.N_CLASSES, ModelConfig.HIDDEN_OUTPUT)
			model.load_state_dict( torch.load("./models/weights/catdog_weights.pt", map_location=torch.device('cpu'), weights_only=True))
			return model
		except Exception as e:
			LOGGER.log.error(f"Load model failed")
			LOGGER.log.error(f"Error: {e}")
			return None


	def transforms_(self, pil_img):
		return self.self_transforms(pil_img)

	def create_transform(self):
			transform = transforms.Compose(
					[
							transforms.Resize((CatDogDataConfig.IMG_SIZE,CatDogDataConfig.IMG_SIZE)),
							transforms.Grayscale(num_output_channels=3),
							transforms.ToTensor(),
							transforms.Normalize(
									CatDogDataConfig.NORMALIZE_MEAN,
									CatDogDataConfig.NORMALIZE_STD
							)
					]
			)
			return transform