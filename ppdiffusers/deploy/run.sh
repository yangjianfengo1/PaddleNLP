cd ../../
python setup.py install
cd ppdiffusers
python setup.py install
cd deploy
python apply_loara_paddle_static.py
# python export_model.py --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 --output_path stable-diffusion-v1-5 --height=512 --width=512 > temp.txt