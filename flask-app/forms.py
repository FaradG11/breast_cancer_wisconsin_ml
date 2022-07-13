from flask_wtf import FlaskForm
from wtforms import TextAreaField
from wtforms.validators import DataRequired, Length


class JsonForm(FlaskForm):
    input_data = TextAreaField(
        label="Field for JSON",
        name="input_data",
        validators=[
            DataRequired(),
        ]
    )
