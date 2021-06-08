def exit_app(self):
    sys.exit(app.exec_())


def enable_randomize(self, x):
    self.numTokens.setEnabled(bool(x))
    self.numTokens.setText("")


def change_window(self, index):
    if index == 0:
        self.Mask.raise_()
        self.Mask.setEnabled(True)

        self.Clasificador.setEnabled(False)
        self.Preguntas.setEnabled(False)

    elif index == 1:
        self.Clasificador.raise_()
        self.Clasificador.setEnabled(True)

        self.Preguntas.setEnabled(False)
        self.Mask.setEnabled(False)

    elif index == 2:
        self.Preguntas.raise_()
        self.Preguntas.setEnabled(True)

        self.Clasificador.setEnabled(False)
        self.Mask.setEnabled(False)


def run_app(self):
    index = self.modeloSeleccion.currentIndex()

    if index == 0:
        predicted_token = maskModel.evaluate_mask(self.fraseEntrada.text(), 4)
        self.textoPredicho.setText(predicted_token)

    elif index == 1:
        pass

    elif index == 2:
        answers = qaModel.answer_questions(questions=[self.pregunta1.toPlainText(), self.pregunta2.toPlainText()],
                                           abstract=self.texto.toPlainText())
        self.respuesta1.setText(answers[0])
        self.respuesta2.setText(answers[1])