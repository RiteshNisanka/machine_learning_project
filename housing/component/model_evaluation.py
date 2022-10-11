from housing.logger import logging
from housing.exception import HousingException
from housing.entity.config_entity import ModelEvaluationConfig
from housing.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact,ModelTrainerArtifact,ModelEvaluationArtifact
from housing.constant import *
import numpy as np
import os
import sys
from housing.util.util import write_yaml_file, read_yaml_file, load_object,load_data
from housing.entity.model_factory import evaluate_regression_model




class ModelEvaluation:

    def __init__(self, model_evaluation_config: ModelEvaluationConfig,
                 data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_artifact: DataValidationArtifact,
                 model_trainer_artifact: ModelTrainerArtifact):
        try:
            logging.info(f"{'>>' * 30}Model Evaluation log started.{'<<' * 30} ")
            self.model_evaluation_config = model_evaluation_config
            self.model_trainer_artifact = model_trainer_artifact
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_artifact = data_validation_artifact
        except Exception as e:
            raise HousingException(e, sys) from e

    def get_best_model(self):
        try:
            model = None
            model_evaluation_file_path = self.model_evaluation_config.model_evaluation_file_path

            if not os.path.exists(model_evaluation_file_path):
                write_yaml_file(file_path=model_evaluation_file_path,
                                )
                return model
                #executing the file for the first time, there is no best model in the evaluation.yaml
                #so we will return none, we will create a file path and return model means return none  
            model_eval_file_content = read_yaml_file(file_path=model_evaluation_file_path)
            # read the yaml file and store it in content

            model_eval_file_content = dict() if model_eval_file_content is None else model_eval_file_content
             # if content is empty, create a null dict(), else store the content info in content itself.
             #just checking, if no content , then empty dict.

            if BEST_MODEL_KEY not in model_eval_file_content:
                return model
                #if file is there but best model is not there, then also return none (model)

            model = load_object(file_path=model_eval_file_content[BEST_MODEL_KEY][MODEL_PATH_KEY])
            return model
        except Exception as e:
            raise HousingException(e, sys) from e

    def update_evaluation_report(self, model_evaluation_artifact: ModelEvaluationArtifact):
        # i will call this function only when my train model is better than the model in the production
        try:
            eval_file_path = self.model_evaluation_config.model_evaluation_file_path
            model_eval_content = read_yaml_file(file_path=eval_file_path)
            model_eval_content = dict() if model_eval_content is None else model_eval_content
            
            
            previous_best_model = None
            if BEST_MODEL_KEY in model_eval_content:
                previous_best_model = model_eval_content[BEST_MODEL_KEY]
                # if there is best model, we are just naming it as previous_best_model and we will 
                #compare it with new model

            logging.info(f"Previous eval result: {model_eval_content}")
            eval_result = {
                BEST_MODEL_KEY: {
                    MODEL_PATH_KEY: model_evaluation_artifact.evaluated_model_path,
                }
                # eval model has the trained model,we will compare it with the previous model
            }

            if previous_best_model is not None:
                model_history = {self.model_evaluation_config.time_stamp: previous_best_model}
                if HISTORY_KEY not in model_eval_content:
                    history = {HISTORY_KEY: model_history}
                    eval_result.update(history)
                else:
                    model_eval_content[HISTORY_KEY].update(model_history)
                    

            model_eval_content.update(eval_result)
            #model_eval_content is a dict, eval_result is also a dict.
            #if content_ev_cnt is empty, then data of eval _result will be updated in model_eval_content dict
            logging.info(f"Updated eval result:{model_eval_content}")
            write_yaml_file(file_path=eval_file_path, data=model_eval_content)

        except Exception as e:
            raise HousingException(e, sys) from e

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
            trained_model_file_path = self.model_trainer_artifact.trained_model_file_path
            trained_model_object = load_object(file_path=trained_model_file_path)

            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            schema_file_path = self.data_validation_artifact.schema_file_path

            train_dataframe = load_data(file_path=train_file_path,
                                                           schema_file_path=schema_file_path,
                                                           )
            test_dataframe = load_data(file_path=test_file_path,
                                                          schema_file_path=schema_file_path,
                                                          )
            schema_content = read_yaml_file(file_path=schema_file_path)
            target_column_name = schema_content[TARGET_COLUMN_KEY]

            # target_column
            logging.info(f"Converting target column into numpy array.")
            train_target_arr = np.array(train_dataframe[target_column_name])
            test_target_arr = np.array(test_dataframe[target_column_name])
            logging.info(f"Conversion completed target column into numpy array.")

            # dropping target column from the dataframe
            logging.info(f"Dropping target column from the dataframe.")
            train_dataframe.drop(target_column_name, axis=1, inplace=True)
            test_dataframe.drop(target_column_name, axis=1, inplace=True)
            logging.info(f"Dropping target column from the dataframe completed.")

            model = self.get_best_model()

            if model is None:
                logging.info("Not found any existing model. Hence accepting trained model")
                model_evaluation_artifact = ModelEvaluationArtifact(evaluated_model_path=trained_model_file_path,
                                                                    is_model_accepted=True)
                self.update_evaluation_report(model_evaluation_artifact)
                logging.info(f"Model accepted. Model eval artifact {model_evaluation_artifact} created")
                return model_evaluation_artifact
# if model is none, then only it will return mode_eval_artifact, means it will accept the trained model
#further below comands will not execute but if there is some model, then line 131 to 136 will not execute
#it will strt from below steps
            model_list = [model, trained_model_object]
            # model is existing model and trained_model_object is the new model

            metric_info_artifact = evaluate_regression_model(model_list=model_list,
                                                               X_train=train_dataframe,
                                                               y_train=train_target_arr,
                                                               X_test=test_dataframe,
                                                               y_test=test_target_arr,
                                                               base_accuracy=self.model_trainer_artifact.model_accuracy,
                                                               )
# here comparsion is taking place between 2 models which are in the model list. passed as parameter in this
# modelREgression funtion and will give the best model. 
# we are doing comparision between best model(model) and new model (traininig model) and give new best model
            logging.info(f"Model evaluation completed. model metric artifact: {metric_info_artifact}")

            if metric_info_artifact is None:
                # metric info can be none, may be both models are not able to achieve base accuracy
                response = ModelEvaluationArtifact(is_model_accepted=False,
                                                   evaluated_model_path=trained_model_file_path
                                                   )
                logging.info(response)
                return response

            if metric_info_artifact.index_number == 1:
                model_evaluation_artifact = ModelEvaluationArtifact(evaluated_model_path=trained_model_file_path,
                                                                    is_model_accepted=True)
                self.update_evaluation_report(model_evaluation_artifact)
                # trained model is better than the model in the production, thts why i updated the report
                logging.info(f"Model accepted. Model eval artifact {model_evaluation_artifact} created")

            else:
                logging.info("Trained model is no better than existing model hence not accepting trained model")
                #  no need to update the report as trained model is not accepted , already existing production model 
                # is the best one.
                model_evaluation_artifact = ModelEvaluationArtifact(evaluated_model_path=trained_model_file_path,
                                                                    is_model_accepted=False)
            return model_evaluation_artifact
        except Exception as e:
            raise HousingException(e, sys) from e
# this is how we implemented RETRAINING of model - means we are comparing our trained model with the model
# in the production, which is the best model till now, if this(trained) model is better than the production 
# model, then we consider it as a best one. 
    def __del__(self):
        logging.info(f"{'=' * 20}Model Evaluation log completed.{'=' * 20} ")