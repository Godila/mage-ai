from mage_ai.api.presenters.BasePresenter import BasePresenter


class ProjectPresenter(BasePresenter):
    default_attributes = [
        'emr_config',
        'features',
        'features_defined',
        'features_override',
        'help_improve_mage',
        'latest_version',
        'name',
        'openai_api_key',
        'openai_base_url',
        'openai_model',
        'ai_mode',
        'pipelines',
        'platform_settings',
        'project_uuid',
        'projects',
        'remote_variables_dir',
        'repo_path',
        'root_project',
        'settings',
        'spark_config',
        'version',
        'workspace_config_defaults',
    ]
