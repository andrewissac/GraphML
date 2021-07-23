import jsonpickle
from os import path
from pathlib import Path

class JsonSerializable():
    def ToJsonString(self):
        jsonpickle.set_encoder_options('json', sort_keys=False, indent=4)
        return jsonpickle.encode(self)

    def SaveToJsonfile(self, outputPath, outputFilename):
        jsonString = self.ToJsonString()
        Path(outputPath).mkdir(parents=True, exist_ok=True)
        file = Path(path.join(outputPath, outputFilename))
        if file.is_file():
            userInput = input(f'File {outputFilename} already exists! Do you want to overwrite the file? (y/n)\n')
            userInput = userInput.lower()
            if userInput == 'y':
                with open(file, 'w') as jsonFile:
                    jsonFile.write(jsonString)
                print(f"File {outputFilename} overwritten.")
            else:
                print(f'File {outputFilename} was not overwritten.')
                pass
        else:
            with open(file, 'w') as jsonFile:
                jsonFile.write(jsonString)


    @classmethod
    def LoadFromJsonString(cls, jsonString):
        return jsonpickle.decode(jsonString)

    @classmethod
    def LoadFromJsonfile(cls, filePath):
        with open(filePath, 'r') as jsonFile:
            jsonString = jsonFile.read()
            return cls.LoadFromJsonString(jsonString)