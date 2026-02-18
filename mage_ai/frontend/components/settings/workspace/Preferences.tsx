import { useCallback, useEffect, useMemo, useState } from 'react';
import { useMutation } from 'react-query';

import Button from '@oracle/elements/Button';
import Divider from '@oracle/elements/Divider';
import Flex from '@oracle/components/Flex';
import FlexContainer, {
  JUSTIFY_SPACE_BETWEEN_PROPS,
} from '@oracle/components/FlexContainer';
import Headline from '@oracle/elements/Headline';
import Link from '@oracle/elements/Link';
import Panel from '@oracle/components/Panel';
import ProjectType, { FeatureUUIDEnum, ProjectRequestPayloadType } from '@interfaces/ProjectType';
import SetupSection, { SetupSectionRow } from '@components/shared/SetupSection';
import Spacing from '@oracle/elements/Spacing';
import Text from '@oracle/elements/Text';
import TextInput from '@oracle/elements/Inputs/TextInput';
import ToggleSwitch from '@oracle/elements/Inputs/ToggleSwitch';
import Tooltip from '@oracle/components/Tooltip';
import api from '@api';
import useProject from '@utils/models/project/useProject';
import { ContainerStyle } from './index.style';
import { Edit } from '@oracle/icons';
import { ICON_SIZE_SMALL } from '@oracle/styles/units/icons';
import { LOCAL_TIMEZONE_TOOLTIP_PROPS, storeLocalTimezoneSetting } from './utils';
import { PADDING_UNITS, UNITS_BETWEEN_SECTIONS } from '@oracle/styles/units/spacing';
import { capitalizeRemoveUnderscoreLower } from '@utils/string';
import { ignoreKeys } from '@utils/hash';
import { onSuccess } from '@api/utils/response';
import { useError } from '@context/Error';

type PreferencesProps = {
  cancelButtonText?: string;
  contained?: boolean;
  header?: any;
  onCancel?: () => void;
  onSaveSuccess?: (project: ProjectType) => void;
  rootProject?: boolean;
};

function Preferences({
  cancelButtonText,
  contained,
  header,
  onCancel,
  onSaveSuccess,
  rootProject: rootProjectUse,
}: PreferencesProps) {
  const [showError] = useError(null, {}, [], {
    uuid: 'settings/workspace/preferences',
  });
  const [projectAttributes, setProjectAttributes] = useState<ProjectType>(null);
  const [editingOpenAIKey, setEditingOpenAIKey] = useState<boolean>(false);
  const [editingOpenAIBaseUrl, setEditingOpenAIBaseUrl] = useState<boolean>(false);

  const {
    fetchProjects,
    project: projectInit,
    projectPlatformActivated,
    rootProject,
  } = useProject();

  const project = useMemo(() => rootProjectUse ? rootProject : projectInit, [
    projectInit,
    rootProject,
    rootProjectUse,
  ]);

  const {
    name: projectName,
    openai_api_key: openaiApiKey,
    openai_base_url: openaiBaseUrl,
    project_uuid: projectUUID,
  } = project || {};

  const isDemoApp = useMemo(() =>
    typeof window !== 'undefined' && window.location.hostname === 'demo.mage.ai',
    [],
  );

  useEffect(() => {
    if (!projectAttributes) {
      setProjectAttributes(project);
    }
  }, [project, projectAttributes]);

  const [updateProjectBase, { isLoading: isLoadingUpdateProject }]: any = useMutation(
    api.projects.useUpdate(projectName),
    {
      onSuccess: (response: any) => onSuccess(
        response, {
          callback: ({ project: p }) => {
            fetchProjects();
            setProjectAttributes(p);
            setEditingOpenAIKey(false);
            setEditingOpenAIBaseUrl(false);
            storeLocalTimezoneSetting(p?.features?.[FeatureUUIDEnum.LOCAL_TIMEZONE]);

            if (onSaveSuccess) {
              onSaveSuccess?.(p);
            }
          },
          onErrorCallback: (response, errors) => showError({
            errors,
            response,
          }),
        },
      ),
    },
  );

  const updateProject = useCallback((payload: ProjectRequestPayloadType) => updateProjectBase({
    project: {
      ...payload,
      root_project: rootProjectUse,
    },
  }), [
    rootProjectUse,
    updateProjectBase,
  ]);

  const el = (
    <>
      {header}

      <Panel noPadding>
        <Spacing p={PADDING_UNITS}>
          <Spacing mb={1}>
            <Headline level={5}>
              Project name
            </Headline>
          </Spacing>

          <Text default monospace>
            {projectName}
          </Text>
        </Spacing>

        <Divider light />

        <Spacing p={PADDING_UNITS}>
          <Spacing mb={1}>
            <Headline level={5}>
              Project UUID
            </Headline>
          </Spacing>

          <Text default={!!projectUUID} monospace muted={!projectUUID}>
            {projectUUID || 'Not required'}
          </Text>
        </Spacing>

        <Divider light />

        <Spacing p={PADDING_UNITS}>
          <FlexContainer
            alignItems="center"
            justifyContent="space-between"
          >
            <Flex flexDirection="column">
              <Spacing mb={1}>
                <Headline level={5}>
                  Help improve Mage
                </Headline>
              </Spacing>

              <Text default>
                Please contribute usage statistics to help improve the developer experience
                for you and everyone in the community. Learn more <Link
                  href="https://docs.mage.ai/contributing/statistics/overview"
                  openNewWindow
                >
                  here
                </Link>.
              </Text>
            </Flex>

            <Spacing mr={PADDING_UNITS} />

            <ToggleSwitch
              checked={projectAttributes?.help_improve_mage}
              compact
              id="help_improve_mage_toggle"
              onCheck={() => setProjectAttributes(prev => ({
                ...prev,
                help_improve_mage: !projectAttributes?.help_improve_mage,
              }))}
            />
          </FlexContainer>
        </Spacing>

        {/*<Divider light />

        <Spacing p={PADDING_UNITS}>
          <Spacing mb={1}>
            <Headline level={5}>
              Automatically generate block names
            </Headline>
          </Spacing>

          <FlexContainer alignItems="center">
            <Checkbox
              checked={automaticallyNameBlocks}
              label="Use randomly generated names for blocks created in the future"
              onClick={() => {
                setAutomaticallyNameBlocks(!automaticallyNameBlocks);
                set(LOCAL_STORAGE_KEY_AUTOMATICALLY_NAME_BLOCKS, !automaticallyNameBlocks);
              }}
            />
          </FlexContainer>
        </Spacing>*/}
      </Panel>

      <Spacing mt={UNITS_BETWEEN_SECTIONS} />

      <SetupSection
        description="Global settings that are applied to all pipelines in this project."
        title="Pipeline settings"
      >
        <SetupSectionRow
          description="Every time a trigger is created or updated in this pipeline, automatically persist it in code."
          title="Save triggers in code automatically"
          toggleSwitch={{
            checked: !!projectAttributes?.pipelines?.settings?.triggers?.save_in_code_automatically,
            onCheck: (valFunc: (val: boolean) => boolean) => setProjectAttributes(prev => ({
              ...prev,
              pipelines: {
                ...prev?.pipelines,
                settings: {
                  ...prev?.pipelines?.settings,
                  triggers: {
                    ...prev?.pipelines?.settings?.triggers,
                    save_in_code_automatically: valFunc(
                      prev?.pipelines?.settings?.triggers?.save_in_code_automatically,
                    ),
                  },
                },
              },
            })),
          }}
        />
      </SetupSection>

      <Spacing mt={UNITS_BETWEEN_SECTIONS} />

      <Panel noPadding overflowVisible>
        <Spacing p={PADDING_UNITS}>
          <Spacing mb={1}>
            <Headline level={5}>
              Features&nbsp;
              <Link
                bold
                href="https://docs.mage.ai/development/project/features"
                largeSm
                openNewWindow
              >
                (docs)
              </Link>
            </Headline>
          </Spacing>

          {Object.entries(ignoreKeys(projectAttributes?.features, [
            FeatureUUIDEnum.CODE_BLOCK_V2,
            FeatureUUIDEnum.COMMAND_CENTER,
            FeatureUUIDEnum.COMPUTE_MANAGEMENT,
            FeatureUUIDEnum.CUSTOM_DESIGN,
            FeatureUUIDEnum.DBT_V2,
            FeatureUUIDEnum.GLOBAL_HOOKS,
            FeatureUUIDEnum.NOTEBOOK_BLOCK_OUTPUT_SPLIT_VIEW,
          ]) || {}).map(([k, v], idx) => {
            const overrideFromRootProject = projectPlatformActivated
              && !rootProjectUse
              && project?.features_override
              && k in project?.features_override;

            return (
              <Spacing
                key={k}
                mt={idx === 0 ? 0 : 1}
              >
                <FlexContainer
                  alignItems="center"
                >
                  <Flex flex={1}>
                    <ToggleSwitch
                      disabled={overrideFromRootProject}
                      checked={!!v}
                      compact
                      onCheck={() => setProjectAttributes(prev => ({
                        ...prev,
                        features: {
                          ...projectAttributes?.features,
                          [k]: !v,
                        },
                      }))}
                    />

                    <Spacing mr={PADDING_UNITS} />

                    <Flex>
                      <Text default={!v} monospace>
                        {capitalizeRemoveUnderscoreLower(k)}
                      </Text>

                      {k === FeatureUUIDEnum.LOCAL_TIMEZONE &&
                        <Spacing ml={1}>
                          <Tooltip
                            {...LOCAL_TIMEZONE_TOOLTIP_PROPS}
                          />
                        </Spacing>
                      }
                    </Flex>
                  </Flex>

                  {overrideFromRootProject && (
                    <Text monospace muted small>
                      overridden
                    </Text>
                  )}
                </FlexContainer>
              </Spacing>
            );
          })}
        </Spacing>
      </Panel>

      <Spacing mt={UNITS_BETWEEN_SECTIONS} />

      <Panel noPadding>
        <Spacing p={PADDING_UNITS}>
          <Spacing mb={1}>
            <Headline level={5}>
              AI provider
            </Headline>
          </Spacing>

          <Spacing mb={2}>
            <Text default small>
              Select <Text bold inline monospace small>open_ai</Text> for OpenAI, or{' '}
              <Text bold inline monospace small>open_ai_compatible</Text> for any OpenAI-compatible
              provider (Ollama, Groq, Together AI, LM Studio, etc.).
            </Text>
          </Spacing>

          <Spacing mb={2}>
            <Text muted small>
              AI mode
            </Text>
            <select
              onChange={e => setProjectAttributes(prev => ({
                ...prev,
                ai_mode: e.target.value,
              }))}
              style={{ marginTop: 4, width: '100%', padding: '6px 8px' }}
              value={projectAttributes?.ai_mode || 'open_ai'}
            >
              <option value="open_ai">open_ai (OpenAI)</option>
              <option value="open_ai_compatible">open_ai_compatible (Ollama, Groq, etc.)</option>
              <option value="hugging_face">hugging_face (Hugging Face)</option>
            </select>
          </Spacing>

          <Spacing mb={2}>
            {(openaiApiKey && !editingOpenAIKey)
              ?
                <FlexContainer {...JUSTIFY_SPACE_BETWEEN_PROPS} >
                  <Text default monospace>
                    API key: ********
                  </Text>
                  <Button
                    iconOnly
                    onClick={() => setEditingOpenAIKey(true)}
                    secondary
                    title="Edit"
                  >
                    <Edit size={ICON_SIZE_SMALL} />
                  </Button>
                </FlexContainer>
              :
                <TextInput
                  disabled={isDemoApp}
                  label={isDemoApp ? 'Entering API key is disabled on demo' : 'API key'}
                  monospace
                  onChange={e => setProjectAttributes(prev => ({
                    ...prev,
                    openai_api_key: e.target.value,
                  }))}
                  primary
                  setContentOnMount
                  value={projectAttributes?.openai_api_key || ''}
                />
            }
          </Spacing>

          <Spacing mb={2}>
            {(openaiBaseUrl && !editingOpenAIBaseUrl)
              ?
                <FlexContainer {...JUSTIFY_SPACE_BETWEEN_PROPS} >
                  <Text default monospace>
                    Base URL: {openaiBaseUrl}
                  </Text>
                  <Button
                    iconOnly
                    onClick={() => setEditingOpenAIBaseUrl(true)}
                    secondary
                    title="Edit"
                  >
                    <Edit size={ICON_SIZE_SMALL} />
                  </Button>
                </FlexContainer>
              :
                <TextInput
                  disabled={isDemoApp}
                  label="Base URL (leave blank for default OpenAI)"
                  monospace
                  onChange={e => setProjectAttributes(prev => ({
                    ...prev,
                    openai_base_url: e.target.value,
                  }))}
                  placeholder="https://api.openai.com/v1"
                  primary
                  setContentOnMount
                  value={projectAttributes?.openai_base_url || ''}
                />
            }
          </Spacing>

          <TextInput
            disabled={isDemoApp}
            label="Model (leave blank for default gpt-4o)"
            monospace
            onChange={e => setProjectAttributes(prev => ({
              ...prev,
              openai_model: e.target.value,
            }))}
            placeholder="gpt-4o"
            primary
            setContentOnMount
            value={projectAttributes?.openai_model || ''}
          />
        </Spacing>
      </Panel>

      <Spacing mt={UNITS_BETWEEN_SECTIONS} />

      <FlexContainer alignItems="center">
        <Button
          id="save-project-settings"
          loading={isLoadingUpdateProject}
          onClick={() => {
            const updateProjectPayload: ProjectRequestPayloadType = {
              ai_mode: projectAttributes?.ai_mode,
              features: projectAttributes?.features,
              help_improve_mage: projectAttributes?.help_improve_mage,
              openai_api_key: projectAttributes?.openai_api_key,
              openai_base_url: projectAttributes?.openai_base_url,
              openai_model: projectAttributes?.openai_model,
              pipelines: projectAttributes?.pipelines,
            };
            if (project?.help_improve_mage === true
              && projectAttributes?.help_improve_mage === false
            ) {
              updateProjectPayload.deny_improve_mage = true;
            }
            updateProject(updateProjectPayload);
          }}
          primary
        >
          Save project settings
        </Button>

        {onCancel && (
          <>
            <Spacing mr={PADDING_UNITS} />

            <Button
              onClick={onCancel}
              secondary
            >
              {cancelButtonText || 'Cancel'}
            </Button>
          </>
        )}
      </FlexContainer>
    </>
  );

  if (contained) {
    return (
      <ContainerStyle>
        {el}
      </ContainerStyle>
    );
  }

  return el;
}

export default Preferences;
