
class Registry(object):
    class_name_dict = {}

    @classmethod
    def register_model(cls, cls_name):

        def wrap(model_cls):

            cls.class_name_dict[cls_name] = model_cls

            return model_cls

        return wrap

    @classmethod
    def get_model_class(cls, name):
        return cls.class_name_dict.get(name, None)

registry = Registry()